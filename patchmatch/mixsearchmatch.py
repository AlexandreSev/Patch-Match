# -*- coding: utf-8 -*-
"""
Function to find the best match
"""
import numpy as np
from scipy.linalg import norm
from multiprocessing import JoinableQueue as Queue
import multiprocessing
import threading
import time
import cython

def create_kernel_local(xc, yc, psi):
    """
    Compute the weight of pixels according to their distance to one pixel
    """
    result = np.zeros((psi, psi))
    for i in range(psi):
        for j in range(psi):
            result[i,j] = float((i-xc)**2 + (j-yc)**2) / 2
    m = np.max(result)        
    result = 1 - result.astype(np.float) / m
    return result

def create_kernel_patch(is_compared, psi):
    """
    Create a matrix with the weight of pixels, according to their distance to the hole
    """
    result = np.zeros(is_compared.shape)
    counter = 0
    for i in range(is_compared.shape[0]):
        for j in range(is_compared.shape[1]):
            if is_compared[i,j] == 0:
                result += create_kernel_local(i, j, psi)
                counter += 1
    result = result.astype(np.float) / counter
    return result * is_compared

def distance(img1, img2, gauss, param=2):
    """
    Compute the distance between two patchs
    """
    temp = img1.reshape((img1.shape[0], img1.shape[1], -1))
    shape = temp.shape[2]
    result = 0

    for i in range(shape):
        result += norm((img1[:, :, i] - img2[:, :, i]) * gauss, 2)
    return result

def findbestmatch(patch, img, is_compared, patch_x, patch_y, psi, search_area_size=200, 
		verbose=True, rotation=True):
    """
    Find the best match without parallelisation.
    """

    if rotation:
        r_max = 4
    else:
        r_max = 1

    result = 9999
    gauss = create_kernel_patch(is_compared[:, :, 0], psi)
    counter = 0
    for i in range(max(0, int(patch_x - search_area_size)), int(patch_x + search_area_size + 1)):
            for j in range(max(0, int(patch_y - search_area_size)), int(patch_y + search_area_size + 1)):
                for r in range(r_max):
                    counter +=1
                    if (counter % 50000 == 0) & verbose:
                        print("Counter %s"%counter)
                    patch_comp = patch * is_compared
                    if (i<psi/2) | (j<psi/2) | (i>img.shape[0]-psi/2-1) | (j>img.shape[1]-psi/2-1):
                        pass
                    else:
                        img_comp = img[i - psi/2: i + psi/2+1, j - psi/2: j + psi/2+1].copy()
                        img_comp = np.rot90(img_comp, r)
                        if np.all(img_comp[:, :, 0] != -1):
                            img_comp *= is_compared
                            if distance(img_comp, patch_comp, is_compared[:, :, 0], gauss) < result:
                                result = distance(img_comp, patch_comp, is_compared[:, :, 0], gauss)
                                result_x = i
                                result_y = j
                                result_r = r
    return result_x, result_y, result_r

def find_best_match(patch, marged_img, is_compared, patch_x, patch_y, psi, search_area_size=100, 
		verbose=True, rotation=True, multi_processing=True, n_processes=None, n_batch=None):
    """
    Return the part of the picture with the lowest distance to the patch.
    """
    if multi_processing:
        result_x, result_y, result_r = findbestmatchmultiprocess(patch, marged_img, is_compared, patch_x, 
            patch_y, psi, search_area_size=search_area_size, verbose=verbose, rotation=rotation, 
            n_processes=n_processes, n_batch=n_batch)
    else:
        result_x, result_y, result_r = findbestmatch(patch, marged_img, is_compared, patch_x, 
        	patch_y, psi, search_area_size=search_area_size, verbose=verbose, rotation=rotation)
    return result_x, result_y, result_r



""" 
A partir d'ici, se sont les fonctions servant à parralléliser ce calcul
"""

class Worker_process(multiprocessing.Process):
    """
    Define a custon worker with two more attributes
    """

    def __init__(self, args=()):
    	multiprocessing.Process.__init__(self)
    	self.args = args
    	self.result = (0, 0, 0)
    	self.distance = 99999

    def run(self):
    	img_queue, result_queue, gauss, patch, psi, size_x, size_y, r_max, verbose = self.args
    	self.worker_function(img_queue, result_queue, gauss, patch, psi, size_x, size_y, r_max, verbose)
    	return

    def worker_function(self, img_queue, result_queue, gauss, patch, psi, size_x, size_y, 
                        r_max=1, verbose=True):
    	"""
    	While the queue with all works is not empty, the worker will take the last element, Compute
    	the calcul and store this element if the distance to the patch is the smallest he found.
    	"""
        while True:
            if img_queue.qsize() == 0:
                break
            else:
                if verbose:
                    print(self.name + ": " + str(img_queue.qsize()) + " elements left" )
                img_temp, decal_i, decal_j = img_queue.get()
                for i in range(psi, img_temp.shape[0] - psi + 1):
                    for j in range(psi, img_temp.shape[1] - psi + 1):
                        for r in range(r_max):
                            if (i<psi/2) | (j<psi/2) | (i>size_x-psi/2-1) | (j>size_y-psi/2-1):
                                pass
                            else:
                                img_comp = img_temp[i - psi/2: i + psi/2+1, j - psi/2: j + psi/2+1].copy()
                                img_comp = np.rot90(img_comp, r)
                                if np.all(img_comp[:, :, 0] != -1):
                                    if distance(img_comp, patch, gauss) < self.distance:
                                        self.distance = distance(img_comp, patch, gauss)
                                        self.result = (decal_i + i, decal_j + j, r)
        result_queue.put((self.distance, self.result))
        return

def create_queue(main_queue, patch_x, patch_y, r_max, search_area_size):
    """
    Create the queue with all centers of patch we have to examine
    """
    for i in range(max(0, int(patch_x - search_area_size)), int(patch_x + search_area_size + 1)):
            for j in range(max(0, int(patch_y - search_area_size)), int(patch_y + search_area_size + 1)):
            	for r in range(r_max):
                	main_queue.put((i, j, r))

def get_result(result_queue):
    """
    Return the result with the lowest distance
    """
    distance, result_xyr = result_queue.get()
    while result_queue.qsize() != 0:
    	distance_temp, result_xyr_temp = result_queue.get()
    	if distance_temp < distance:
            result_xyr = result_xyr_temp
            distance = distance_temp
    return result_xyr


def findbestmatchmultiprocess(patch, img, is_compared, patch_x, patch_y, psi=9, search_area_size=200, 
		verbose=True, rotation=False, n_processes=None, n_batch=None):
    """
    Find the best match with multi processing.
    """
    if rotation:
    	r_max = 4
    else:
        r_max = 1

    img_queue = Queue()
    result_queue = Queue()

    if n_processes is None:
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = n_processes

    if n_batch is None:
        num_batch = multiprocessing.cpu_count()
    elif n_batch<num_processes:
        num_batch = num_processes
    else:
        num_batch = n_batch

    gauss = create_kernel_patch(is_compared[:, :, 0], psi)

    batch_size = (int(patch_x + search_area_size + 1) - max(0, int(patch_x - search_area_size))) \
                                    // num_batch

    i_min = max(0, int(patch_x - search_area_size - psi))

    j_min, j_max = max(0, int(patch_y - search_area_size)), int(patch_y + search_area_size + 1)
    size_x = img.shape[0]
    size_y = img.shape[1]

    jobs = []

    for i in range(num_batch):
        if i == num_batch - 1:
            i_max = int(patch_x + search_area_size + 1)
        else:
            i_max = (i+1) * batch_size + psi + 1
        img_temp = img[max(0, i_min + i * batch_size - psi): min(img.shape[0], i_max), 
                        max(0, j_min - psi): min(img.shape[1], j_max + psi + 1), :]
        decal_i = max(0, i_min + i * batch_size - psi)
        decal_j = max(0, j_min - psi)
        img_queue.put((img_temp, decal_i, decal_j))
        if i < num_processes:
            p = Worker_process(args=(img_queue, result_queue, gauss, patch, psi, size_x, size_y, 
                            r_max, verbose))
            jobs.append(p)
            p.start()
        

    if verbose:
    	print '*** Main thread waiting'

    for job in jobs:
        job.join()

    if verbose:
    	print '*** Done'

    time.sleep(1)

    return get_result(result_queue)