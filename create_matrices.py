# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:52:22 2023

@author: Manuel
"""

import numpy as np
import json
import math


with open('./hotdog/transforms_test.json', 'r') as f:
    data = json.load(f)
    
camera_fov_x = data['camera_angle_x']
image_width = 800  # In pixels
image_height = 800  # In pixels


# Loop over the frames and extract the camera pose for each view
for i, frame in enumerate(data['frames']):
    # Extract the rotation angle and translation from the transform matrix
    translation = np.array(frame["transform_matrix"])[:3, 3]
    
    rotation_matrix = np.array(frame["transform_matrix"])[:3, :3]
    s = frame["rotation"]
    # Compute the K matrix using the given camera FOV
    fx = image_width / 2*(math.tan((camera_fov_x / 2)))
    fy = fx  # Assuming square pixels and no skew
    cx = image_width / 2
    cy = image_height / 2
    K = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]])

    # Save the camera pose to a file
    np.savetxt(f'./hotdog/views_test/cam{i:06d}_k.txt', K)
    np.savetxt(f'./hotdog/views_test/cam{i:06d}_r.txt', rotation_matrix)
    np.savetxt(f'./hotdog/views_test/cam{i:06d}_t.txt', translation)