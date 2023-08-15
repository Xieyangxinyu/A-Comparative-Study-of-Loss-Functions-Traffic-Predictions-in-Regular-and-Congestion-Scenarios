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


def masked_focal_mae_loss(preds, labels, null_val=np.nan, activate='sigmoid', beta=.2, gamma=1):
    mask = get_mask(labels, null_val=null_val)

    loss = torch.abs(preds-labels)
    loss *= (torch.tanh(beta * torch.abs(preds - labels))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(preds - labels)) - 1) ** gamma
    
    return get_masked_loss(loss, mask)


def masked_focal_mse_loss(preds, labels, null_val=np.nan, activate='sigmoid', beta=.2, gamma=1):
    mask = get_mask(labels, null_val=null_val)

    loss = (preds-labels) ** 2
    loss *= (torch.tanh(beta * torch.abs(preds - labels))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(preds - labels)) - 1) ** gamma
    
    return get_masked_loss(loss, mask)


def masked_mape(preds, labels, null_val=np.nan):
    mask = get_mask(labels, null_val=null_val)
    loss = torch.abs(preds-labels)/labels
    return get_masked_loss(loss, mask)


def masked_bmc_loss_1(pred, labels, null_val, noise_var = 1.0):
    pred = pred.transpose(1, 2)
    pred = pred.flatten(0, 1)
    labels = labels.transpose(1, 2)
    labels = labels.flatten(0, 1)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    pred = pred * mask
    logits = -(pred.unsqueeze(1) - labels.unsqueeze(0)).pow(2).sum(2) / (2 * noise_var)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())     # contrastive-like loss
    loss = torch.nan_to_num(loss)
    return loss


def masked_bmc_loss_9(pred, labels, null_val, noise_var = 9.0):
    pred = pred.transpose(1, 2)
    pred = pred.flatten(0, 1)
    labels = labels.transpose(1, 2)
    labels = labels.flatten(0, 1)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    pred = pred * mask
    logits = -(pred.unsqueeze(1) - labels.unsqueeze(0)).pow(2).sum(2) / (2 * noise_var)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())     # contrastive-like loss
    loss = torch.nan_to_num(loss)
    return loss

def masked_kirtosis(preds, labels, null_val=np.nan):
    mask = get_mask(labels, null_val=null_val)

    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    aux_loss = (preds-labels) ** 2
    aux_loss = aux_loss * mask
    aux_loss = torch.where(torch.isnan(aux_loss), torch.zeros_like(aux_loss), aux_loss)

    mean = torch.mean(aux_loss)
    std  = torch.std(aux_loss)
    aux_loss = ((aux_loss - mean) / std) ** 4
    loss = loss + 0.01 * aux_loss
    loss = torch.mean(loss)
    loss = torch.nan_to_num(loss)
    return loss


def masked_huber(preds, labels, null_val=np.nan, beta=1.):
    mask = get_mask(labels, null_val=null_val)

    l1_loss = torch.abs(preds-labels)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)

    return get_masked_loss(loss, mask)


def masked_Gumbel(preds, labels, null_val=np.nan, gamma = 1.1):
    mask = get_mask(labels, null_val=null_val)

    l2_loss = (preds-labels) ** 2
    loss = ((1 - torch.exp(-l2_loss)) ** gamma) * l2_loss

    return get_masked_loss(loss, mask)

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse
