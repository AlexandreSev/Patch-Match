# -*- coding: utf-8 -*-
"""
Main file to run patch match
"""
from utils import *
from priority import *
from searchmatch import *
from mixingpatch import *
import numpy as np
np.random.seed(42)

def patch_match(img_ini, psi_min=21, psi_max=71, search_area_size=200, multi_processing=True, 
		 rotation=False, verbose=True, create_hole=True, x_hole=(120,180), y_hole=(180,230), 
		 blur=True):
	"""
	Fill a hole with the algorithm Patch_Match
	"""

	marged_img = preprocess_picture(img_ini, psi_max, verbose=verbose, create_hole=create_hole, 
			x_hole=x_hole, y_hole=y_hole)

	confidence = create_confidence_matrix(marged_img)
	nb_iter = 0
	cut_edges = {}

	criteria, contour = find_good_contour(marged_img)

	while criteria:

		if (nb_iter%10 == 0) & verbose:
			show_picture(marged_img, "Iteration %r"%nb_iter)
		else:
			print("Iteration " + str(nb_iter))
		nb_iter += 1

		psi = np.random.randint(psi_min, psi_max) / 2 * 2 + 1

		if verbose:
			print("psi = %s"%psi)

		patch_x, patch_y, confidence_used = get_working_area(confidence, contour, marged_img, psi)

		patch = get_patch(marged_img, patch_x, patch_y, psi)
		is_compared = (patch != -1)

		if verbose:
			show_picture(patch, "Area %r"%nb_iter)

		result_x, result_y, result_r = find_best_match(patch, marged_img, is_compared, patch_x, 
				patch_y, psi=psi, search_area_size=search_area_size, verbose=verbose, rotation=rotation, 
				multi_processing=multi_processing)

		true_patch = get_patch(marged_img, result_x, result_y, psi, result_r)

		if verbose:
			show_picture(true_patch, "Best match %r"%nb_iter)

		applied_patch = get_mixed_patch(marged_img, result_x, result_y, result_r, patch_x, patch_y, psi, 
				cut_edges, true_patch, blur=blur)

		if verbose:
			show_picture(applied_patch, "Applied Patch %r"%nb_iter)

		img[np.int(patch_x - psi/2 + 0.5): np.int(patch_x + psi/2+1 + 0.5), 
					np.int(patch_y - psi/2 + 0.5): np.int(patch_y + psi/2+1 + 0.5)] = applied_patch

		confidence = update_confidence(confidence, confidence_used, patch_x, patch_y, psi)

		criteria, contour = find_good_contour(marged_img)
	return marged_img
	

if __name__ == "__main__":

	from scipy.misc import face
	import time
	time_init=time.time()
	img = face()
	result = patch_match(img, psi_min=50, psi_max=73, verbose=False)
	print("Done in %s seconds"%(time.time()-time_init))
	show_picture(result, "Result")
