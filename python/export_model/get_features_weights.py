import numpy as np
import keras
import keras.layers as layers

def export_conv2d_weights_and_biases(layer, path,  name):
  weights = np.array(layer.weights[0])
  biases  = np.array(layer.weights[1]) 

  with open(path+'/weights/'+name+'_w.txt', 'w') as f:
    for ofm in range(weights.shape[3]):   
      for ifm in range(weights.shape[2]):   
        for row in range(weights.shape[0]):  
          for col in range(weights.shape[1]): 
            f.write(str(weights[row][col][ifm][ofm]))
            if (col < (weights.shape[1]-1)):
              f.write(',')
            else:
              f.write('\n')

  with open(path+'/biases/'+name+'_b.txt', 'w') as f:
    for ofm in range(biases.shape[0]):  
      f.write(str(biases[ofm]))
      if (ofm < (biases.shape[0]-1)):
        f.write(',')


def export_dense_weights_and_biases(layer, path,  name):
  weights = np.array(layer.weights[0])
  biases  = np.array(layer.weights[1]) 

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


def export_all_w_b(model, path):
  for layer in model.layers:
    # name = path+'/'+layer.name
    if isinstance(layer, keras.layers.Conv2D):    
      export_conv2d_weights_and_biases(layer, path, layer.name)
    elif isinstance(layer, keras.layers.Dense):
      export_dense_weights_and_biases(layer, path, layer.name)