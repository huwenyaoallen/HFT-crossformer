import numpy as np
import torch.nn as nn
import torch
import scipy.stats as st
import matplotlib.pyplot as plt
def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def RMSPE(pred, true):
    mse = np.mean(((true - pred) / (true+1e-6)) ** 2)
    return np.sqrt(mse)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rmspe= RMSPE(pred, true)
    return mae,mse,rmse,mape,mspe,rmspe

def PCC(pred,true):
    pred = np.concatenate(pred, axis=0)
    true = np.concatenate(true, axis=0)
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    #    Plot the indices on the x-axis and the values on the y-axis
    ax.plot(range(len(pred)), pred, label='Predicted')
    ax.plot(range(len(true)), true, label='True')

    # Set the labels and title
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Predicted vs. True Values')
    ax.legend()

    # Save the figure
    plt.savefig('pred_true_plot.png')
    ic =st.spearmanr(pred,true)
    pred = torch.tensor(pred)   
    true = torch.tensor(true)
    cos=nn.CosineSimilarity(dim=0, eps=1e-6)
    #print(pred)
    #print(true)
    return cos(pred-pred.mean(dim=0,keepdim=True),true-true.mean(dim=0,keepdim=True)).mean(),ic



