import torch
from typing import Dict, List
import cv2
from nds.core import View
import numpy as np
from PIL import Image


def scale_image(image, out_size, device):
    """ Scale an image to a desired size """
    image = image.cpu().detach().numpy().astype(np.float32)
    resized_image = cv2.resize(image, dsize=(out_size[0], out_size[1]), interpolation=cv2.INTER_LINEAR)
    return torch.FloatTensor(resized_image).to(device)

def normal_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], L, compare_size, device):
    """ Compute the normal term as the combination of the L (L1 or L2) and angular loss
    
    Args:
        views (List[View]): Views with normals
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'normal' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """
    loss = 0.0
    for view, gbuffer in zip(views, gbuffers):
        vnormal = torch.FloatTensor(np.array(Image.open(view.normal).convert('RGB'))).to(view.device)
        vnormal = scale_image(vnormal, (compare_size,compare_size), device)
        #apply mask to the view normal
        #vnormal = vnormal * gbuffer["mask"] + (1-gbuffer["mask"]) * 255

        normal = (0.5*(gbuffer["normal"] @ view.camera.R.T + 1)) * gbuffer["mask"] + (1-gbuffer["mask"])
        normal *= 255
        #normal = torch.clamp(gbuffer["normal"], min=0)
        #normal *= 255*gbuffer["mask"]
        normal = scale_image(normal, (compare_size,compare_size), device)

        loss += L(vnormal, normal) + ((1 - torch.cosine_similarity(vnormal, normal))).mean()
        #loss += L(view.normal, normal) + ((1 - torch.cosine_similarity(view.normal, normal))).mean()

    return loss / len(views)