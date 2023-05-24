import torch
from typing import Dict, List
import cv2
from nds.core import View

def normal_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], L1 = torch.nn.L1Loss()):
    """ Compute the normal term as $$$the mean difference between the original masks and the rendered masks.

    implementar L1 loss e ngular loss
    
    Args:
        views (List[View]): Views with normals
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'normal' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = 0.0
    for view, gbuffer in zip(views, gbuffers):
        normal = gbuffer["normal"]*255*gbuffer["mask"]
        loss += L1(view.normal, normal) + ((1 - torch.cosine_similarity(view.normal, normal))).mean()
    return loss / len(views)