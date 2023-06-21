import torch
from typing import Dict, List
import cv2
from nds.core import View
import numpy as np
from PIL import Image

def depth_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], L = torch.nn.L1Loss()):
    """ Compute the normal term as the combination of the L (L1 or L2) and angular loss
    
    Args:
        views (List[View]): Views with normals
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'normal' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = 0.0
    for view, gbuffer in zip(views, gbuffers):
        if view.scale_factor:
            vdepth = view.scale_depth(view.depth)
        else:
            vdepth = torch.FloatTensor(np.array(Image.open(view.depth).convert('RGB'))).to(view.device)
        vdepth*=view.mask
        #normal = torch.clamp(gbuffer["normal"], min=0)
        depth = gbuffer["depth"]*gbuffer["mask"]
        loss += L(vdepth, depth) + ((1 - torch.cosine_similarity(vdepth, depth))).mean()
        #loss += L(view.normal, normal) + ((1 - torch.cosine_similarity(view.normal, normal))).mean()
    return loss / len(views)