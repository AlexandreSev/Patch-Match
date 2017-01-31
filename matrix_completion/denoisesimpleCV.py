from SimpleCV import Image, Display, Color, Camera
import time 
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import math

threshold = 0.9
cam = Camera(0)
disp = Display((512,512))

def noise(X, PropObserved):
	Noisyimage = X.copy()
	n = int(np.ceil(PropObserved*X.shape[0]))
	for i in range(Noisyimage.shape[1]): #on parcourt les colonnes
		TobeRemoved = np.random.choice(range(X.shape[0]),n, replace=False)
		Noisyimage[TobeRemoved,i] = -1
	return Noisyimage

def svd_shrink(X, tau):
    """
    Apply the shrinkage operator to the singular values obtained from the SVD of X.
    The parameter tau is used as the scaling parameter to the shrink function.
    Returns the matrix obtained by computing U * shrink(s) * V where 
        U are the left singular vectors of X
        V are the right singular vectors of X
        s are the singular values as a diagonal matrix
    """
    U,s,V = np.linalg.svd(X, full_matrices=False)
    return np.dot(U, np.dot(np.diag(shrink(s, tau)), V))
def shrink(X, tau):
    """
    Apply the shrinkage operator the the elements of X.
    Returns V such that V[i,j] = max(abs(X[i,j]) - tau,0).
    """
    V = np.copy(X).reshape(X.size)
    for i in xrange(V.size):
        V[i] = math.copysign(max(abs(V[i]) - tau, 0), V[i])
        if V[i] == -0:
            V[i] = 0
    return V.reshape(X.shape)
def frobeniusNorm(X):
    """
    Evaluate the Frobenius norm of X
    Returns sqrt(sum_i sum_j X[i,j] ^ 2)
    """
    accum = 0
    V = np.reshape(X,X.size)
    for i in xrange(V.size):
        accum += abs(V[i] ** 2)
    return np.sqrt(accum)
def L1Norm(X):
    """
    Evaluate the L1 norm of X
    Returns the max over the sum of each column of X
    """
    return max(np.sum(X,axis=0))
def converged(M,L,S, verbose = True, tol=10e-6):
    """
    A simple test of convergence based on accuracy of matrix reconstruction
    from sparse and low rank parts
    """
    error = frobeniusNorm(M - L - S) / frobeniusNorm(M)
    if verbose:
        print ("error =", error)
    return error <= tol
def robust_pca(M, maxiter=10, verbose = True):
    """ 
    Decompose a matrix into low rank and sparse components.
    Computes the RPCA decomposition using Alternating Lagrangian Multipliers.
    Returns L,S the low rank and sparse components respectively
    """
    L = np.zeros(M.shape)
    S = np.zeros(M.shape)
    Y = np.zeros(M.shape)
    mu = (M.shape[0] * M.shape[1]) / (4.0 * L1Norm(M))
    lamb = max(M.shape) ** -0.5
    iter = 0
    while not converged(M,L,S, verbose) and iter < maxiter:
        if iter % 10 == 0:
            print(iter)
        L = svd_shrink(M - S - (mu**-1) * Y, mu)
        S = shrink(M - L + (mu**-1) * Y, lamb * mu)
        Y = Y + mu * (M - L - S)
        iter+= 1
    return L,S
def NMF(X, r, lbda=1., Max_iter=200):
    for i in range(Max_iter):
        W = np.random.randint(low=0,high=255,size=(X.shape[0], r))
        H = np.linalg.solve(W.T.dot(W), W.T.dot(X))
        H[H<0] = 0
        W = np.linalg.solve(H.dot(H.T), H.dot(X.T)).T
        W[W<0] = 0
    return W.dot(H)

def project_orth(X, Z):
    Y = X.copy()
    """
    X the predicted image
    Z is the observed image
    """
    Y[Z>0] = 0
    return Y
def project(X, Z):
    Y = X.copy()
    """
    See slide 11 from http://web.stanford.edu/~hastie/TALKS/SVD_hastie.pdf
    X the predicted image
    Z is the observed image
    """
    Y[Z==-1] = 0
    return Y

iter = 0 # faire quelque chose de mieux en utilisant l'heure

stackpred = np.zeros((480,640,2)) # la taille de mon display 480, 640
while True and iter<20:
        
        #previous = cam.getImage() 
        time.sleep(0.01) 
        current = cam.getImage()
        #img.drawText("Paul, virons ce texte", 40,40, fontsize=60, color=Color.RED) 
        #diff = current - previous
        pic = current.getNumpy()[:,:,0].T
        #matrix = diff.getNumpy()
        #mean = matrix.mean()
        noisyPic = noise(pic, 0.7)
        denoised = NMF(noisyPic, 50 ,Max_iter=20)

        if iter % 2 == 0:
        	stackpred[:,:,0] = noisyPic+project(denoised, noisyPic)
        elif iter % 2 == 1:
        	stackpred[:,:,1] = noisyPic+project(denoised, noisyPic)

       	current.show()
       	plt.ion()
       	plt.figure(3)
       	plt.title("OverWholeBetter Image")
        #print noisyPic.shape
        #print project(np.mean(stackpred, axis=2), noisyPic).shape
        mybuffer = np.zeros((noisyPic.shape[0],noisyPic.shape[1],2))
        mybuffer[:,:,1] = project(np.mean(stackpred, axis=2), noisyPic)
        mybuffer[:,:,0] = noisyPic
        #print np.mean(noisyPic, project(np.mean(stackpred, axis=2), noisyPic))
       	plt.imshow(np.mean(mybuffer, axis=2), cmap='gray')
        plt.draw()
        plt.show()

        plt.figure(4)
        reconstructed = np.maximum(mybuffer[:,:,0], mybuffer[:,:,1])
        plt.imshow(reconstructed, cmap='gray')
        plt.draw()
        plt.show()
        
        plt.figure(5)
        reconstructed2 = np.maximum(np.mean(mybuffer, axis=2), mybuffer[:,:,1])
        plt.imshow(reconstructed2, cmap="gray")
       	plt.draw()
       	plt.show()
        
       	print time.asctime() + "distance reconstructed " + str(np.abs(np.linalg.norm(pic) - np.linalg.norm(reconstructed)))
        print time.asctime() + "distance mean " + str(np.abs(np.linalg.norm(pic) - np.linalg.norm(np.mean(mybuffer, axis=2))))
        print time.asctime() + "distance reconstructed2 " + str(np.abs(np.linalg.norm(pic) - np.linalg.norm(reconstructed2)))

        iter+=1
        #if mean >= threshold:
        #        print time.asctime()+" Motion Detected"