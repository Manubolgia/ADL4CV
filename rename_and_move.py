# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:37:47 2023

@author: Manuel
"""

import os
import shutil

source_folder = './hotdog/test'
destination_folder = './hotdog/views_test'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate over files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file has the correct format (r_i)
    if filename.startswith('r_') and not filename.endswith('0029.png'):
        # Extract the file number i without the extension
        file_number_without_extension = os.path.splitext(filename[2:])[0]
        
        # Convert the file number to an integer
        image_number = int(file_number_without_extension)
        
        # Define the new filename using the new naming convention
        new_filename = f'cam{image_number:06d}'
        
        # Get the file extension (e.g., '.jpg', '.png')
        file_extension = os.path.splitext(filename)[1]
        
        # Combine the new filename with its extension
        new_filename_with_extension = new_filename + file_extension
        
        # Define the source and destination file paths
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, new_filename_with_extension)
        
        # Copy the file to the destination folder with the new name
        shutil.copyfile(source_path, destination_path)

print("Files have been copied and renamed.")

