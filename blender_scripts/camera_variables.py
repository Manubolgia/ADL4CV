# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:59:29 2023

@author: Manuel
"""
import json
import bpy
         

fp = 'Destination folder'


# Camera Data to store in JSON file
out_data = {
    'Focal_length_mm': bpy.data.objects['Camera'].data.lens,
    'sensor_H': bpy.data.objects['Camera'].data.sensor_height,
    'sensor_W': bpy.data.objects['Camera'].data.sensor_width,
    'shift_x': bpy.data.objects['Camera'].data.shift_x,
    'shift_y': bpy.data.objects['Camera'].data.shift_y,
    'image_H': bpy.context.scene.render.resolution_y,
    'image_W': bpy.context.scene.render.resolution_x,
    'pixel_aspect_x': bpy.context.scene.render.pixel_aspect_x,
    'pixel_aspect_y': bpy.context.scene.render.pixel_aspect_y
}

with open(fp + '/' + 'camera_variables.json', 'w') as out_file:
    json.dump(out_data, out_file, indent=4)
