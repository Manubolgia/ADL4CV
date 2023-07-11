import os
import shutil

source_folder = ''
destination_folder = ''

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate over files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file has the correct format (r_i)
    if filename.endswith('_depth.png'): #and 'depth' not in filename:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        
        # Copy the file to the destination folder with the new name
        os.remove(source_path)

print("Files have been copied and renamed.")

