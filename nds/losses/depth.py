import torch
from typing import Dict, List
import cv2
from nds.core import View
import numpy as np
from PIL import Image
from torch import nn
import math
import cv2


# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy    

def scale_image(image, out_size, device):
    """ Scale an image to a desired size """
    image = image.cpu().detach().numpy().astype(np.float32)
    resized_image = cv2.resize(image, dsize=(out_size[0], out_size[1]), interpolation=cv2.INTER_LINEAR)
    return torch.FloatTensor(resized_image).to(device)

def depth_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], compare_size, device):
    """ Compute the depth term 
    Args:
        
    """
    loss = 0.0
    depth_loss_function = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

    for view, gbuffer in zip(views, gbuffers):

        depth_gt = torch.FloatTensor(np.array(Image.open(view.depth).convert('RGB')))[:,:,0:1]
        depth_gt = scale_image(depth_gt, (compare_size,compare_size), device)
        depth_gt = process_depth(depth_gt)

        depth_pred = scale_image(gbuffer["depth"].squeeze(), (compare_size,compare_size), device)
        depth_pred = process_depth(depth_pred)

        mask = (gbuffer['mask'] > 0.5) & (view.mask > 0.5)
        mask = scale_image(mask, (compare_size,compare_size), device)

        
        loss += depth_loss_function(
            depth_pred.reshape(1, compare_size, compare_size),
            depth_gt.reshape(1, compare_size, compare_size),
            mask.reshape(1, compare_size, compare_size)
            ) 
    
    
    return loss / len(views)

def process_depth(image_B):
    """
    This function processes depth ground truth images from the synthetic Nerf dataset.
    It modifies the range of depth values to the 0-255 scale and inverts the depth values.
    
    Parameters:
    image_B: tensor, pytorch tensor of the depth
    
    Return:
    image_B_modified: pytorch tensor, modified depth image with nearest point intensity = 0 and furthest =255
    
    """

    # Ensure the image array is of type float
    image_B = image_B.to(dtype=torch.float64)
    
    # Define the new maximum value for the image
    Ra = 255.0

    image_min = np.amin(image_B.cpu().numpy())
    image_max = np.amax(image_B.cpu().numpy())
    # Compute the original range of pixel values
    Rb = image_max - image_min

    # Adjust the pixel value range of the image
    image_B_modified = image_B - image_min 
    image_B_modified *= Ra / Rb

    # Convert the float array to uint8
    image_B_modified = image_B_modified.to(dtype=torch.uint8)

    # Invert the depth values
    return image_B_modified

