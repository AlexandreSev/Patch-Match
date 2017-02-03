# -*- coding: utf-8 -*-

"""
Functions to find the area with the highest priority
"""

import numpy as np
from skimage import measure

def find_good_contour(img):
	"""
	Find contours of the hole
	"""
	contours = measure.find_contours(img[:,:,0], 0.8)
	for contour in contours:
	    pt = contour[len(contour)/2]
	    pt = [int(i) for i in pt]
	    if np.any(img[pt[0]-1:pt[0]+2, pt[1]-1:pt[1]+2]==[-1,-1,-1]) :
	        contour = np.concatenate((np.array([contour[0]]), contour, np.array([contour[-1]])))
	        contour = contour+0.5
	        return True, contour
	return False, []

def confidence_coefficient(confidence, pt, psi):
	"""
	Return the coefficient of confidence in the point pt given the confidence matrix
	"""
	result = 0
	for i in range(psi):
		for j in range(psi):
		    result += confidence[np.int(pt[0] + i - (psi-1)/2+0.5), np.int(pt[1] + j - (psi-1)/2 +0.5)]
	return float(result)/psi**2

def get_confidence_vector(confidence, omega, psi):
	"""
	Compute the confidence term in all points of omega
	"""
	confidence_vector = [confidence_coefficient(confidence, pt, psi) for pt in omega][1:-1]
	return confidence_vector

def get_normal_vector(omega):
	"""
	Given a contour omega, return normal vector in all interior points.
	"""
	tangeantes = [(omega[i+1][0] - omega[i-1][0],omega[i+1][1] - omega[i-1][1]) for i in range(1, len(omega)-1)]
	normals = [(-j/np.sqrt(i*i+j*j), i/np.sqrt(i*i+j*j)) for i,j in tangeantes]
	return normals 

def pooling(temp):
	"""
	Return the maximum absolute value
	"""
	return np.max(np.abs(temp))

def get_isophote(omega, img, psi):
	"""
	Given a contour omega, return the isophote in each point
	"""
	isophote = []
	for pt in omega:
	    x = pt[0]
	    y = pt[1]
	    temp = img[np.int(x - psi/2): np.int(x + psi/2+1), np.int(y - psi/2): np.int(y + psi/2+1)]
	    grad = np.gradient(np.ma.masked_where(temp == -1, temp))
	    isophote.append((pooling(grad[0]), pooling(grad[1])))
	return isophote

def get_data_vector(omega, img, psi, alpha=255.):
	"""
	Compute the data term for all points given a contour.
	"""
	normal = get_normal_vector(omega)
	isophote = get_isophote(omega[1:-1], img, psi)
	result = [np.abs(normal[i][0] * isophote[i][0] + normal[i][1] * isophote[i][1])/alpha for i in range(len(isophote))]
	return result

def get_priority_vector(confidence_vector, omega, img, psi):
	"""
	Compute the priority term in all points of omega
	"""
	data_vector = get_data_vector(omega, img, psi)
	priority_vector = [data_vector[i] * confidence_vector[i] for i in range(len(data_vector))]
	return priority_vector

def get_working_area(confidence, contour, img, psi):
	"""
	Return coordonates of the point with the highest priority
	"""
	omega=[(i[0], i[1]) for i in contour]

	confidence_vector = get_confidence_vector(confidence, omega, psi)
	priority_vector = get_priority_vector(confidence_vector, omega, img, psi)
	m = max(priority_vector)

	indices = [ i for i,j in enumerate(priority_vector) if j==m]
	indice = indices[len(indices)/2]

	patch_x, patch_y = omega[indice+1]

	patch_x = int(patch_x + 0.5)
	patch_y = int(patch_y + 0.5)

	return patch_x, patch_y, confidence_vector[indice]
