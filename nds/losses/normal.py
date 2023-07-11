import torch
from typing import Dict, List
import cv2
from nds.core import View
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def scale_image(image, out_size, device):
    """ Scale an image to a desired size """
    image = image.cpu().detach().numpy().astype(np.float32)
    resized_image = cv2.resize(image, dsize=(out_size[0], out_size[1]), interpolation=cv2.INTER_LINEAR)
    return torch.FloatTensor(resized_image).to(device)

def normal_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], L, compare_size, normal_format, device):
""" 
    Compute the normal term as the combination of the L (L1 or L2) and angular loss. This function 
    calculates a normal-related loss based on ground truth and predicted normal maps for each view 
    in the input list. The loss is calculated using the loss function `L` and a cosine similarity 
    comparison for angular loss.
    
    Args:
        views (List[View]): A list of View objects containing ground truth normal and mask information.
        gbuffers (List[Dict[str, torch.Tensor]]): A list of dictionaries where each dictionary
                                                   represents a g-buffer, containing predicted normal maps and 
                                                   masks from the neural network.
        L (Callable): A function for comparing the masks or generally a set of pixels. This function 
                      is applied to calculate the primary part of the loss.
        compare_size (int): The size (height and width) to which normal maps and masks should be 
                            scaled before loss computation. The compare_size should be an integer value.
        normal_format (str): Specifies the format of the normal. Should be either 'omni' or 'NERF'. 
                             Different formats imply different preprocessing steps.
        device (str): The device on which computations will be performed. Should be either 'cpu' or 'cuda'.
        
    Returns:
        float: The average normal loss for all views.
    
    Notes:
        This function performs the following steps for each view and corresponding g-buffer:
        1. Load the ground truth normal from the view and scales it according to the compare_size.
        2. Depending on the 'normal_format', applies a specific process to the predicted normal from 
           the g-buffer and scales it according to the compare_size.
        3. Calculates loss using the provided L function and adds angular loss computed as 
           cosine dissimilarity between predicted and ground truth normals.
        4. Returns the average loss for all views.
        
        The function assumes that normal maps are three-channel images, and all RGB channels are used 
        when loading RGB images as normal maps.
    """
    loss = 0.0
    for view, gbuffer in zip(views, gbuffers):

        vnormal = torch.FloatTensor(np.array(Image.open(view.normal).convert('RGB')))*view.mask.cpu()
        vnormal += (1-view.mask.cpu()) * 255
        vnormal = scale_image(vnormal, (compare_size,compare_size), device)

        if normal_format == 'NERF':
            normal = torch.clamp(gbuffer["normal"], min=0) * gbuffer["mask"] + (1-gbuffer["mask"])
            normal *= 255
        else:
            normal = (0.5*(gbuffer["normal"] @ view.camera.R.T + 1)) * gbuffer["mask"] + (1-gbuffer["mask"])
            normal *= 255
        
        normal = scale_image(normal, (compare_size,compare_size), device)

        loss += L(vnormal, normal) + ((1 - torch.cosine_similarity(vnormal, normal))).mean()


    return loss / len(views)