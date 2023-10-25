import numpy as np
import torch
from torch import nn
import keras.layers as layers

def export_conv2d_weights_and_biases(layer, path,  name):
  weights = np.array(layer.weight.detach().numpy())
  biases  = np.array(layer.bias.detach().numpy()) 

  with open(path+'/weights/'+name+'_w.txt', 'w') as f:
    for ofm in range(weights.shape[0]):   
      for ifm in range(weights.shape[1]):   
        for row in range(weights.shape[2]):  
          for col in range(weights.shape[3]): 
            f.write(str(weights[ofm][ifm][row][col]))
            if (col < (weights.shape[3]-1)):
              f.write(',')
            else:
              f.write('\n')

  with open(path+'/biases/'+name+'_b.txt', 'w') as f:
    for ofm in range(biases.shape[0]):  
      f.write(str(biases[ofm]))
      if (ofm < (biases.shape[0]-1)):
        f.write(',')


def export_dense_weights_and_biases(layer, path,  name):
  weights = np.array(layer.weight.detach().numpy())
  biases  = np.array(layer.bias.detach().numpy()) 

  with open(path+'/weights/'+name+'_w.txt', 'w') as f:
    for ifm in range(weights.shape[0]):   
      for ofm in range(weights.shape[1]):   
        f.write(str(weights[ifm][ofm]))
        if (ofm < (weights.shape[1]-1)):
          f.write(',')
        else:
          f.write('\n')

  with open(path+'/biases/'+name+'_b.txt', 'w') as f:
    for ofm in range(biases.shape[0]):  
      f.write(str(biases[ofm]))
      if (ofm < (biases.shape[0]-1)):
        f.write(',')

def export_batchnorm2d_scales_and_biases(layer, path,  name):
  gamma = np.array(layer.weight.detach().numpy())
  beta  = np.array(layer.bias.detach().numpy())
  mean    = np.array(layer.running_mean.detach().numpy())
  var     = np.array(layer.running_var.detach().numpy())
  eps     = layer.eps

  scale = []
  bias  = []
  for i in range(len(gamma)):
    scale.append(gamma[i] / np.sqrt(var[i] + eps))
    bias.append(beta[i] - scale[i] * mean[i])

  with open(path+'/weights/'+name+'_s.txt', 'w') as f:
    for ofm in range(len(scale)):    
      f.write(str(scale[ofm]))
      f.write('\n')

  with open(path+'/biases/'+name+'_b.txt', 'w') as f:
    for ofm in range(len(bias)):    
      f.write(str(bias[ofm]))
      f.write('\n')


def export_all_w_b(model, path):
  names = []
  for name, module in model.named_children():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
      names.append(name)
  
  li = 0
  for layer in model.modules():
    if isinstance(layer, nn.Conv2d):    
      export_conv2d_weights_and_biases(layer, path, names[li])
      li += 1
    elif isinstance(layer, nn.Linear):
      export_dense_weights_and_biases(layer, path, names[li])
      li += 1
    elif isinstance(layer, nn.BatchNorm2d):
      export_batchnorm2d_scales_and_biases(layer, path, names[li])
      li += 1
    
    