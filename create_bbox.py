# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:41:56 2023

@author: Manuel
"""

import cv2
import numpy as np


def read_image(file_path):
    image_rgba = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    image = image_rgba[:,:,3]
    
    return image

def find_bounding_box(frontal_image, lateral_image):
    # Get nonzero pixel indices
    frontal_nonzero = np.nonzero(frontal_image)
    lateral_nonzero = np.nonzero(lateral_image)

    # For the frontal image, x is the second axis and z is the first axis
    x_min, x_max = np.min(frontal_nonzero[1]), np.max(frontal_nonzero[1])
    z_min, z_max = np.min(frontal_nonzero[0]), np.max(frontal_nonzero[0])

    # For the lateral image, y is the second axis and z is shared with the first axis of the frontal image
    y_min, y_max = np.min(lateral_nonzero[1]), np.max(lateral_nonzero[1])

    # Planes equations are simply the constants at the bounds:
    # x = x_min, x = x_max, y = y_min, y = y_max, z = z_min, z = z_max

    # The eight vertices of the bounding box are
    vertices = [
        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
        (x_min, y_max, z_min),
        (x_min, y_max, z_max),
        (x_max, y_min, z_min),
        (x_max, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_max, z_max),
    ]

    return vertices

def find_opposing_vertices(vertices):
    # The vertices are in the following order:
    # (x_min, y_min, z_min),
    # (x_min, y_min, z_max),
    # (x_min, y_max, z_min),
    # (x_min, y_max, z_max),
    # (x_max, y_min, z_min),
    # (x_max, y_min, z_max),
    # (x_max, y_max, z_min),
    # (x_max, y_max, z_max),

    # The opposing vertices are the first and the last ones in this list
    return vertices[0], vertices[-1]

def read_vertices_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    vertices = []
    for line in lines:
        # Split the line into strings representing numbers
        strings = line.split()

        # Convert the strings to floats and pack them into a tuple
        vertex = tuple(float(s) for s in strings)

        # Add the tuple to the list of vertices
        vertices.append(vertex)

    # Return the first and the last vertices in the list
    return vertices[0], vertices[-1]


frontal_syn_pth = "./chair/test/r_150.png"
frontal_syn = read_image(frontal_syn_pth)
lateral_syn_pth = "./chair/test/r_123.png"
lateral_syn = read_image(lateral_syn_pth)
frontal_gt_pth = "C:/Users/Manuel/Documents/GitHub/ADL4CV/data/65_skull/views/cam000046.png"
frontal_gt = read_image(frontal_gt_pth)
lateral_gt_pth = "C:/Users/Manuel/Documents/GitHub/ADL4CV/data/65_skull/views/cam000036.png"
lateral_gt = read_image(lateral_gt_pth)

# Use this code to read the vertices from the file:
bbox_gt_pth = 'C:/Users/Manuel/Documents/GitHub/ADL4CV/data/65_skull/bbox.txt'
bbox_gt =  read_vertices_from_file(bbox_gt_pth)

bbox_syn_p = find_opposing_vertices(find_bounding_box(frontal_syn, lateral_syn))
bbox_gt_p = find_opposing_vertices(find_bounding_box(frontal_gt, lateral_gt))

d_syn_p = (bbox_syn_p[1][0]-bbox_syn_p[0][0],bbox_syn_p[1][1]-bbox_syn_p[0][1],bbox_syn_p[1][2]-bbox_syn_p[0][2])
d_gt_p = (bbox_gt_p[1][0]-bbox_gt_p[0][0],bbox_gt_p[1][1]-bbox_gt_p[0][1],bbox_gt_p[1][2]-bbox_gt_p[0][2])

d_gt_bbox = (bbox_gt[1][0]-bbox_gt[0][0],bbox_gt[1][1]-bbox_gt[0][1],bbox_gt[1][2]-bbox_gt[0][2])

bbox_syn = np.zeros((2,3))

gt_center = ((bbox_gt[1][0]+bbox_gt[0][0])/2,(bbox_gt[1][1]+bbox_gt[0][1])/2,(bbox_gt[1][2]+bbox_gt[0][2])/2)

for k in range(3):
    d_syn_bbox = d_syn_p[k]*(d_gt_bbox[k]/d_gt_p[k])
    bbox_syn[0][k] = 0 - d_syn_bbox/2
    bbox_syn[1][k] = 0 + d_syn_bbox/2
print(bbox_syn,'\n')


np.savetxt("./chair/bbox.txt", bbox_syn)