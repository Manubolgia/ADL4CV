import bpy
from mathutils import Matrix, Vector
import math

# Define a rotation matrix that represents a 180 degree rotation around x-axis
rotation_matrix = Matrix.Rotation(math.pi, 4, 'X')

# Initialize min and max coordinates
min_coord = Vector((float('inf'), float('inf'), float('inf')))
max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

# Iterate over all collections
for collection in bpy.data.collections:
    # Iterate over all objects in the collection
    for obj in collection.objects:
        if obj.type == 'MESH':
            bbox = obj.bound_box
            for corner in bbox:
                # Convert the corner to a vector, apply the rotation, and transform it to world coordinates
                world_corner = obj.matrix_world @ rotation_matrix @ Vector(corner)
                # Update min and max coordinates
                min_coord = Vector((min(c, min_coord[i]) for i, c in enumerate(world_corner)))
                max_coord = Vector((max(c, max_coord[i]) for i, c in enumerate(world_corner)))

# Now min_coord and max_coord are the coordinates of the bounding box corners
# Open the file for writing
with open('Destination folder/bbox.txt', 'w') as file:
    for corner in [min_coord, max_coord]:
        # Format the coordinates as strings and join them with spaces
        line = ' '.join(format(c, 'e') for c in corner) + '\n'
        # Write the line to the file
        file.write(line)
