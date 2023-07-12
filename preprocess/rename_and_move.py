import os
import shutil
import numpy as np
from PIL import Image

def process_image(image_B_path):
    """
    This function processes depth ground truth images from the synthetic Nerf dataset.
    It modifies the range of depth values to the 0-255 scale and inverts the depth values.
    
    Parameters:
    image_B_path: str, path of the depth image to process
    
    Return:
    image_B_modified: numpy.ndarray, modified depth image with nearest point intensity = 0 and furthest =255
    
    """
    # Load image as a numpy array
    image_B = np.array(Image.open(image_B_path))

    # Mask the background (value = 0)
    image_B = np.ma.masked_equal(image_B, 0)

    # Ensure the image array is of type float
    image_B = image_B.astype(np.float64)
    
    # Define the new maximum value for the image
    Ra = 255.0

    # Compute the original range of pixel values
    Rb = np.amax(image_B) - np.amin(image_B)

    # Adjust the pixel value range of the image
    image_B_modified = image_B - np.amin(image_B) 
    image_B_modified *= Ra / Rb

    # Convert the float array to uint8
    image_B_modified = image_B_modified.astype(np.uint8)

    # Invert the depth values
    image_B_modified = 255 - image_B_modified
    
    return image_B_modified

# Set the paths for the source and destination folders
source_folder = ''
destination_folder = ''

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate over files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file has the correct format (r_i.png)
    if filename.startswith('r_') and filename.endswith('.png'):
        # Extract the file number (i) without the extension (.png)
        file_number_without_extension = os.path.splitext(filename)[0] 
        image_number = int(file_number_without_extension.split('_')[1])

        # Set the paths for the source and destination files
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        
        # Define the new filename based on the type of image
        if 'normal' in filename:
            new_filename = f'cam{image_number:06d}_normal'
        elif 'depth' in filename:
            new_filename = f'cam{image_number:06d}_depth'
        else:
            new_filename = f'cam{image_number:06d}'

        # Extract the file extension (.png)
        file_extension = os.path.splitext(filename)[1]

        # Combine the new filename with its extension
        new_filename_with_extension = new_filename + file_extension

        # Set the path for the destination file
        destination_path = os.path.join(destination_folder, new_filename_with_extension)

        # If the image is not a depth image, simply copy it to the destination folder
        if 'depth' not in filename:
            shutil.copyfile(source_path, destination_path)

        # If the image is a depth image, process it before saving to the destination folder
        else:
            # Process the depth image
            image_B_modified = process_image(source_path)
            
            # Save the modified image
            output_path = os.path.join(destination_folder, f'{new_filename_with_extension}')
            Image.fromarray(image_B_modified).save(output_path)

print("Files have been copied, renamed, and processed if necessary.")
