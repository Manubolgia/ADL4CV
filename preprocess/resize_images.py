import os
from PIL import Image
import numpy as np

# Input directory path
input_dir = ''

# Output directory path
output_dir = ''

# Target size
target_size = (800, 800)

# Check if output directory exists, create if not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Variables to keep track of sums and counts of fx, fy, cx, cy
fx_sum = fy_sum = cx_sum = cy_sum = 0
count = 0

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image file
        img = Image.open(os.path.join(input_dir, filename))

        H = img.size[1]
        W = img.size[0]
        
        # Create a square background image
        length = max(img.size)
        background = Image.new('RGBA', (length, length), (255, 255, 255, 0))

        # Calculate the position to paste the image
        pos = ((length - img.size[0]) // 2, (length - img.size[1]) // 2)

        # Paste the image onto the background
        background.paste(img, pos)

        # Resize the image
        img_resized = background.resize(target_size)

        # Save the resized image to the output directory
        img_resized.save(os.path.join(output_dir, filename))

        # Load K matrix from corresponding text file
        text_filename = filename.split('.')[0] + "_k.txt"
        if os.path.exists(os.path.join(input_dir, text_filename)):
            K = np.loadtxt(os.path.join(input_dir, text_filename))

            # Increment sums and count
            fx_sum += K[0, 0]
            fy_sum += K[1, 1]
            cx_sum += K[0, 2]
            cy_sum += K[1, 2]
            count += 1

# Calculate and print mean values
if count > 0:
    print("Mean fx: ", fx_sum / count)
    print("Mean fy: ", fy_sum / count)
    print("Mean cx: ", cx_sum / count)
    print("Mean cy: ", cy_sum / count)

cx_mean = cx_sum / count
cy_mean = cy_sum / count
fx_mean = fx_sum / count
fy_mean = fy_sum / count


shift_x = ((W*0.5) - cx_mean)/W
shift_y = (cy_mean - (H*0.5))/W
constant = fx_mean/W
pixel_aspect = fy_mean/fx_mean

H_new, W_new = target_size

cx_new = W_new * (0.5 - shift_x)
cy_new = H_new*0.5 + W_new*shift_y
fx_new = constant * W_new
fy_new = fx_new * pixel_aspect

K = np.array([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]])

# Loop through each file in the input directory again to save new K matrices
for filename in os.listdir(input_dir):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Save the new K matrix to a text file with the same name as the image
        np.savetxt(os.path.join(output_dir, filename.split('.')[0] + "_k.txt"), K)

print("Image resizing and matrix processing completed.")