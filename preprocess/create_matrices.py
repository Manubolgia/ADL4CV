# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:52:22 2023

@author: Manuel
"""
import os
import numpy as np
import json

# Define the directory where you want to save the files
directory = 'Destination folder'

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

#Load json with Transformation matrices
with open('', 'r') as f:
    data = json.load(f)

#Load camera variables
with open('', 'r') as f:
    camera = json.load(f)

image_width = camera['image_W']  # In pixels
image_height = camera['image_H']  # In pixels

pixel_aspect = camera['pixel_aspect_y'] / camera['pixel_aspect_x']


# Loop over the frames and extract the camera pose for each view
for i, frame in enumerate(data['frames']):
    
    # Define a rotation matrix for a 180-degree rotation about the x-axis
    rotation_180_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotation_4x4 = np.eye(4)  # start with a 4x4 identity matrix
    rotation_4x4[:3, :3] = rotation_180_x
    # Extract the rotation angle and translation from the transform matrix
    T = rotation_4x4 @ np.linalg.inv(np.array(frame["transform_matrix"]))
    
    rotation_matrix = T[:3, :3]
    translation = T[:3, 3]

    
    # Compute the K matrix using the given camera FOV
    f_in_mm = camera['Focal_length_mm']
    sensor_width_in_mm = camera['sensor_W']
    fx = f_in_mm / sensor_width_in_mm * image_width
    fy = fx * pixel_aspect
    cx = image_width * (0.5 - camera['shift_x'])
    cy = image_height * 0.5 + image_width * camera['shift_y']
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Save the camera pose to a file; change the out directory if needed
    np.savetxt(f'{directory}/cam{i:06d}_k.txt', K)
    np.savetxt(f'{directory}/cam{i:06d}_r.txt', rotation_matrix)
    np.savetxt(f'{directory}/cam{i:06d}_t.txt', translation)
