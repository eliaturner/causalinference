# Importing the RNN class and task dataset implementation

import torch
import torch.nn as nn


# Helper function to check convergence
# def has_converged(loss_array, threshold=1e-2):
#     if len(loss_array) > 5:
#         return abs(loss_array[-1] - loss_array[-5]) < threshold
#     return False
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Create a mask that is 1 where y_true is not nan and 0 where y_true is nan
        mask = ~torch.isnan(y_true)

        # Apply the mask to y_pred and y_true
        y_pred_masked = y_pred[mask]
        y_true_masked = y_true[mask]

        # Compute MSE only for non-nan values
        loss = nn.functional.mse_loss(y_pred_masked, y_true_masked)
        return loss
