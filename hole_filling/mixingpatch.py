# -*- coding: utf-8 -*-
"""
Functions to compute the patch to apply
"""

import numpy as np
import networkx as nx
from utils import get_patch
from scipy.linalg import norm

def tuple_add(pt1, pt2):
    """
    compute pt1 + pt2
    """
    return (int(pt1[0] + pt2[0]), int(pt1[1] + pt2[1]))

def tuple_minus(pt1, pt2):
    """
    compute pt1 - pt2
    """
    return (pt1[0] - pt2[0], pt1[1] - pt2[1])

def tuple_in(t, l):
    """
    Find if t or its inverse is in l
    """
    return ((t in l) | ((t[1], t[0]) in l))

def norm3d(img, param):
    """
    Compute the norm for a 3D tensor
    """
    temp = img.reshape((img.shape[0], img.shape[1], -1))
    shape = temp.shape[2]
    result = 0
    for i in range(shape):
        result += norm(img[:, :, i], param)
    return result

def edge_weight(pt1_ini, pt2_ini, old_patch, new_patch, demi=False):   
    """
    Compute the weight of an edge
    """
    if demi:
        diff = tuple_minus(pt2_ini, pt1_ini)

        pt1 = (int(pt1_ini[0]), int(pt1_ini[1]))
        pt2 = tuple_add(pt2_ini, diff)
    else:
        pt1 = (int(pt1_ini[0]), int(pt1_ini[1]))
        pt2 = (int(pt2_ini[0]), int(pt2_ini[1]))

    return norm3d(old_patch[pt1].reshape((1, 1, -1))-new_patch[pt1].reshape((1, 1, -1)), 2) + \
        norm3d(old_patch[pt2].reshape((1, 1, -1))-new_patch[pt2].reshape((1, 1, -1)), 2)

def create_graph(img_ini, pt_ini, new_patch, cut_edges, psi):
    """
    Create the graph associated with the two patches
    """
    img = img_ini.reshape((img_ini.shape[0], img_ini.shape[1], -1))
    pt = (int(pt_ini[0] + 0.5), int(pt_ini[1] + 0.5))
    old_patch = img[pt[0] - psi//2 - 1: pt[0] + psi//2 + 2, pt[1] - psi//2 - 1: pt[1] + psi//2 + 2]
    G=nx.Graph()
    G.add_node("old")
    G.add_node("new")
    for i in range(pt[0] - psi//2, pt[0] + psi//2+1):
        firsti = (i == pt[0] - psi//2)
        lasti = (i == pt[0] + psi//2)
        for j in range(pt[1] - psi//2, pt[1] + psi//2+1):
            firstj = (j == pt[1] - psi//2)
            lastj = (j == pt[1] + psi//2)
            if img[i, j, 0] != -1:
                G.add_node((i, j))
                if not firsti:
                    if ((img[i-1, j, 0] == -1) & ((i-1, j) not in G.nodes())):
                        G.add_node((i-1, j))
                        G.add_edge((i-1, j), 'new', weight=np.inf)
                else:
                    G.add_node((i-1, j))
                    G.add_edge((i-1, j), 'old', weight=np.inf)
                if not lasti:
                    if ((img[i+1, j, 0] == -1) & ((i+1, j) not in G.nodes())):
                        G.add_node((i+1, j))
                        G.add_edge((i+1, j), 'new', weight=np.inf)
                else:
                    G.add_node((i+1, j))
                    G.add_edge((i+1, j), 'old', weight=np.inf)
                if not firstj:
                    if ((img[i, j-1, 0] == -1) & ((i, j-1) not in G.nodes())):
                        G.add_node((i, j-1))
                        G.add_edge((i, j-1), 'new', weight=np.inf)
                else:
                    G.add_node((i, j-1))
                    G.add_edge((i, j-1), 'old', weight=np.inf)
                if not lastj:
                    if ((img[i, j+1, 0] == -1) & ((i, j+1) not in G.nodes())):
                        G.add_node((i, j+1))
                        G.add_edge((i, j+1), 'new', weight=np.inf)
                else:
                    G.add_node((i, j+1))
                    G.add_edge((i, j+1), 'old', weight=np.inf)
                    
                for ibis, jbis in [(i-1, j), (i+1,j), (i, j-1), (i, j+1)]:
                    if (not tuple_in(((i,j), (ibis, jbis)), G.edges())) & ((ibis,jbis) in G.nodes()):
                        if ((i,j), (ibis, jbis)) in cut_edges:
                            idemi, jdemi = (float(i+ibis)//2, float(j+jbis)//2)
                            G.add_node((idemi, jdemi))
                            G.add_edge((idemi,jdemi), 'new', cut_edges)
                            G.add_edge((i, j), (idemi, jdemi), 
                            	weight=edge_weight((i-pt[0]+psi//2+1, j-pt[1]+psi//2+1), 
                                                   (idemi-pt[0]+psi//2+1,jdemi-pt[1]+psi//2+1),
                                                    old_patch, new_patch, demi=True))
                            G.add_edge((ibis, jbis), (idemi, jdemi), 
                            	weight=edge_weight((ibis-pt[0]+psi//2+1, jbis-pt[1]+psi//2+1), 
                                                   (idemi-pt[0]+psi//2+1, jdemi-pt[1]+psi//2+1),
                                                    old_patch, new_patch, demi=True))
                        else:
                            G.add_edge((ibis, jbis), (i,j), 
                            	weight=edge_weight((i-pt[0]+psi//2+1, j-pt[1]+psi//2+1), 
                                                   (ibis-pt[0]+psi//2+1, jbis -pt[1]+psi//2+1),
                                                    old_patch, new_patch))
    return G

def update_cut_edges(cut_edges, new_set):
    """
    Remove edges which have been removed
    """
    new_cut_edges = {}
    for edge in cut_edges:
        node1 = edge[0]
        node2 = edge[1]
        if (node1 not in new_set) | (node2 not in new_set):
            new_cut_edges[edge]= cut_edges[edge]
    return new_cut_edges

def find_cut_edges(G, set_old, new_set, dico_cut_edges):
    """
    Find cut edges during one iteration
    """
    list_just_cut = []
    for edge in G.edges(data=True):
        node1 = edge[0]
        node2 = edge[1]
        if (node1!='old') & (node1!='new') & (node2!='old') & (node2!='new'):
            if ((node1 in set_old) & (node2 not in set_old)) | \
            	((node1 not in set_old) & (node2 in set_old)):
                if np.floor(node1[0]) != node1[0]:
                    node1 = (2 * node1[0] - node2[0], node1[1])
                elif np.floor(node1[1]) != node1[1]:
                    node1 = (node1[0], 2 * node1[1] - node2[1])
                elif np.floor(node2[0]) != node2[0]:
                    node2 = (2 * node2[0] - node1[0], node2[1])
                elif np.floor(node2[1]) != node2[1]:
                    node2 = (node2[0], 2 * node2[1] - node1[1])
                dico_cut_edges[(node1, node2)]= edge[2]["weight"]
                dico_cut_edges[(node2, node1)]= edge[2]["weight"]
                list_just_cut.append((node1, node2))
    return update_cut_edges(dico_cut_edges, new_set), list_just_cut

def clean_mix(old_patch, new_patch, set_old, patch_x, patch_y, psi):
    """
    Mix two patchs without blurring the seam
    """
    applied_patch = new_patch.copy()
    for pt in set_old:
        if (pt!='old'):
            if (np.int(pt[0] - patch_x + psi/2)>=0) & (np.int(pt[0] - patch_x + psi/2)<psi) & \
                (np.int(pt[1] - patch_y + psi/2)>=0) & (np.int(pt[1] - patch_y + psi/2)<psi):
                applied_patch[np.int(pt[0] - patch_x + psi/2), 
                			np.int(pt[1] - patch_y + psi/2)] = \
                            old_patch[np.int(pt[0] - patch_x + psi/2 ), 
                            		  np.int(pt[1] - patch_y + psi/2 )]
    return applied_patch

def blur_mix(img_ini, new_patch, enlarged_new_patch, set_old, just_cut_edges, patch_x, patch_y, psi):
    """
    Mix two patchs and blur the seam
    """

    img = img_ini.reshape((img_ini.shape[0], img_ini.shape[1], -1))
    pt = (int(patch_x + 0.5), int(patch_y + 0.5))
    old_patch = img[pt[0] - psi/2: pt[0] + psi/2 + 1, pt[1] - psi/2: pt[1] + psi/2 + 1]
    enlarged_old_patch = img[pt[0] - psi/2 - 1: pt[0] + psi/2 + 2, pt[1] - psi/2 - 1: pt[1] + psi/2 + 2]
    applied_patch = img[pt[0] - psi/2 - 1: pt[0] + psi/2 + 2, pt[1] - psi/2 - 1: pt[1] + psi/2 + 2]
    applied_patch[1:-1, 1:-1] = clean_mix(old_patch, new_patch, set_old, patch_x, patch_y, psi)

    for edge in just_cut_edges:
        node1 = edge[0]
        node2 = edge[1]

        value11 = np.int(node1[0] - patch_x + psi/2 + 1)
        value12 = np.int(node1[1] - patch_y + psi/2 + 1)
        value21 = np.int(node2[0] - patch_x + psi/2 + 1)
        value22 = np.int(node2[1] - patch_y + psi/2 + 1)

        if node1 in set_old:
            applied_patch[value11, value12] = 0.66 * enlarged_old_patch[value11, value12] + \
            									0.34 * enlarged_new_patch[value11, value12]
            applied_patch[value21, value22] = 0.34 * enlarged_old_patch[value21, value22] + \
            									0.66 * enlarged_new_patch[value21, value22]
        else:
            applied_patch[value11, value12] = 0.34 * enlarged_old_patch[value11, value12] + \
            									0.66 * enlarged_new_patch[value11, value12]
            applied_patch[value21, value22] = 0.66 * enlarged_old_patch[value21, value22] + \
            									0.34 * enlarged_new_patch[value21, value22]

    return applied_patch[1:-1, 1:-1]

def get_mixed_patch(img, true_x, true_y, true_r, patch_x, patch_y, psi, cut_edges, 
	true_patch=None, blur=True):
    """
    Create a graph to find the best patch to apply
    """
    if true_patch is None:
    	true_patch = get_patch(img, true_x, true_y, psi, true_r)

    enlarged_true_patch = get_patch(img, true_x, true_y, psi + 2, true_r)

    G = create_graph(img, (patch_x, patch_y), enlarged_true_patch, cut_edges, psi)
    sets = (nx.minimum_cut(G, 'old', 'new', capacity="weight"))

    if 'old' in sets[1][0]:
        set_old = sets[1][0]
        set_new = sets[1][1]
    else:
        set_old = sets[1][1]
        set_new = sets[1][0]

    cut_edges, just_cut_edges = find_cut_edges(G, set_old, set_new, cut_edges)

    if blur:
    	applied_patch = blur_mix(img, true_patch, enlarged_true_patch, set_old, just_cut_edges, patch_x, patch_y, psi)
    else:
    	applied_patch = clean_mix(patch, true_patch, set_old, patch_x, patch_y, psi)

    return applied_patch
