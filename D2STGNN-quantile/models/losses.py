import torch
import numpy as np
import torch.nn.functional as F


def get_mask(labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def get_masked_loss(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val=np.nan):
    mask = get_mask(labels, null_val=null_val)
    loss = (preds-labels)**2
    return get_masked_loss(loss, mask)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    mask = get_mask(labels, null_val=null_val)
    loss = torch.abs(preds-labels)
    return get_masked_loss(loss, mask)

def masked_mape(preds, labels, null_val=np.nan):
    mask = get_mask(labels, null_val=null_val)
    loss = torch.abs(preds-labels)/labels
    return get_masked_loss(loss, mask)

def quantile_loss(pred, labels, null_val = 0.0):
    
    pred = torch.unbind(pred,2)

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= mask.mean()
    quantiles = [0.025, 0.5, 0.975]
    losses = []
    for i, q in enumerate(quantiles):
        errors =  labels - pred[i]
        errors = errors * mask
        errors[errors != errors] = 0
        losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(0))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return loss


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse