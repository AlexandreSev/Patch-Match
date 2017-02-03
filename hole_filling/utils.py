# -*- coding: utf-8 -*-
"""
Preprocess the picture, and initialise variables
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def preprocess_picture(img_ini, psi_max, verbose=True, create_hole=True, x_hole=(120,180), 
					   y_hole=(180,230)):
	"""
	Create a black marge around the picture. If create_hole, create also a hole with the coordinates 
	given by x_hole and y_hole. 
	"""
	img = img_ini.reshape((img_ini.shape[0], img_ini.shape[1], -1))
	
	if verbose:
		plt.imshow(img, interpolation='none')
		plt.title("Original image")
		plt.show()
		print 'The dim of the image is ' + str(img.shape)
	
	shape = img.shape[2]

	marged_img = (np.zeros((img.shape[0] + 2 * psi_max, img.shape[1] + 2 * psi_max, shape)) + 
		2).astype(int)
	marged_img[psi_max: - psi_max, psi_max:-psi_max] = img.astype(int).copy()

	if create_hole:
		marged_img[x_hole[0]:x_hole[1], y_hole[0]:y_hole[1]] = - np.ones(shape)
	else:
		marged_img[(marged_img[:,:,0]==0) * (marged_img[:,:,1]==0) * (marged_img[:,:,2]==0)] = \
			- np.ones(shape)
	
	#Plot the holed picture
	if verbose:
		plt.imshow(marged_img.astype(np.uint8), interpolation='none', norm=Normalize(vmin=0, vmax=255))
		plt.title("marged_image")
		plt.show()
		print 'The dim of the image is ' + str(marged_img.shape)
	return marged_img

def create_confidence_matrix(marged_img):
	"""
	Initialise the confidence matrix given a picture
	"""
	confidence = np.ones((marged_img.shape[0], marged_img.shape[1]))
	confidence[marged_img[:,:,0] == -1]=0
	confidence[(marged_img[:,:,0] == 2) * (marged_img[:,:,1] == 2) * (marged_img[:,:,2] == 2)]=0
	return confidence

def get_patch(img, x, y, psi, r=0):
	"""
	Given the center x, y, the size psi and the picture img, return the corresponding patch
	"""
	patch = img[np.int(x - psi//2): np.int(x + psi//2 + 1), np.int(y - psi//2): np.int(y + psi//2 + 1)]
	return np.rot90(patch, r)

def show_picture(img, title):
	"""
	Plot the picture
	"""
	plt.imshow(img.astype(np.uint8), norm=Normalize(vmin=0, vmax=255), interpolation='none')
	plt.title(title)
	plt.show()

def update_confidence(confidence, confidence_used, patch_x, patch_y, psi):
	confidence[np.int(patch_x - psi//2): np.int(patch_x + psi//2+1), 
				np.int(patch_y - psi//2): np.int(patch_y + psi//2+1)] = confidence_used
	return confidence
	    