import torch.nn as nn

# Exports a dictionary from a Conv2D layer
def exportConv2D(layer, layers_list):
  ld = {}
  ld['type'] = 'Conv2D'
  ld['name'] = 'conv_'+str(len([l for l in layers_list if l['type']=='Conv2D']))
  ld['R'] = str(38)
  ld['C'] = str(114)
  ld['N'] = str(layer.in_channels)
  ld['M'] = str(layer.out_channels)
  ld['K'] = str(layer.kernel_size[0])
  ld['L'] = str(layer.kernel_size[1])
  ld['stride'] = str(layer.stride[0])
  ld['dilation'] = str(layer.dilation[0])

  padding = [str(layer.padding[0]), str(layer.padding[0]), str(layer.padding[1]), str(layer.padding[1])]
  if layer.padding_mode != 'zeros':
    print('FIX')
    # for i in range(4):
    #   padding[i] = str(int((conf['kernel_size'][0]-1)/2))
  ld['padding'] = padding


  ld['otype'] = 'signedDataT'
  ld['itype'] = 'unsignedDataT' if len(layers_list)==0 else layers_list[-1]['otype']
  ld['activation'] = 'linear'
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['Tm'] = '1'
  ld['Tn'] = '1'
  return ld

# # Exports a dictionary from a MaxPooling2D layer
# def exportMaxPool2D(layer, layers_list):
#   ld = {}
#   ld['type'] = 'Pool2D'
#   ld['name'] = layer.name
#   ld['R'] = str(layer.input_shape[1])
#   ld['C'] = str(layer.input_shape[2])
#   ld['M'] = str(layer.input_shape[3])
#   ld['Tr'] = '1'
#   ld['Tc'] = '1'
#   ld['Tm'] = '1'
#   ld['otype'] = layers_list[-1]['otype']
#   return ld

# Exports a dictionary from a Dense layer
def exportLinear(layer, layers_list):
  ld = {}
  ld['type'] = 'Dense'
  ld['name'] = 'fc_'+str(len([l for l in layers_list if l['type']=='Dense']))
  ld['M'] = str(layer.out_features)
  ld['N'] = str(layer.in_features)

  ld['otype'] = 'signedDataT'
  ld['itype'] = 'unsignedDataT' if len(layers_list)==0 else layers_list[-1]['otype']
  ld['activation'] = 'linear'
  ld['Tn'] = '1'
  ld['Tm'] = '1'
  return ld

def exportReLU(layer, layers_list):
  for i in range(len(layers_list)):
    if layers_list[-1-i]['type'] == 'Conv2D' or layers_list[-1-i]['type'] == 'Dense':
      layers_list[-1-i]['activation'] = 'relu'
      break

def exportSigmoid(layer, layers_list):
  for i in range(len(layers_list)):
    if layers_list[-1-i]['type'] == 'Conv2D' or layers_list[-1-i]['type'] == 'Dense':
      layers_list[-1-i]['activation'] = 'sigmoid'
      break

def exportBatchNorm2d(layer, layers_list):
  ld = {}
  ld['type'] = 'BatchNorm2D'
  ld['name'] = 'batc_'+str(len([l for l in layers_list if l['type']=='BatchNorm2D']))
  ld['R'] = layers_list[-1]['R']
  ld['C'] = layers_list[-1]['C']
  ld['M'] = layers_list[-1]['M']
  ld['Tm'] = '1'
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['otype'] = 'signedDataT'
  ld['itype'] = 'signedDataT'
  return ld

  

# def exportFlatten(layer, layers_list):
#   ld = {}
#   ld['type'] = 'Flatten'
#   ld['name'] = layer.name
#   ld['R'] = layers_list[-1]['R']
#   ld['C'] = layers_list[-1]['C']
#   ld['N'] = layers_list[-1]['M']
#   ld['Tr'] = '1'
#   ld['Tc'] = '1'
#   ld['Tm'] = '1'
#   ld['Tn'] = '1'
#   ld['otype'] = layers_list[-1]['otype']
#   return ld

# Exports the information from a keras model
def export(model):
  layers_list = []
  trainable_layers = 0

  for layer in model.modules():
    # conf = layer.get_config()

    if isinstance(layer, nn.Conv2d):
      trainable_layers += 1
      ld = exportConv2D(layer, layers_list)
      layers_list.append(ld)

    # elif isinstance(layer, keras.layers.MaxPooling2D):
    #   ld = exportMaxPool2D(layer, layers_list)
    #   layers_list.append(ld)

    elif isinstance(layer, nn.BatchNorm2d):
      trainable_layers += 1
      ld = exportBatchNorm2d(layer, layers_list)
      layers_list.append(ld)
      
    elif isinstance(layer, nn.ReLU):
      exportReLU(layer, layers_list)

    elif isinstance(layer, nn.Sigmoid):
      exportSigmoid(layer, layers_list)

    elif isinstance(layer, nn.Linear):
      trainable_layers += 1
      ld = exportLinear(layer, layers_list)
      layers_list.append(ld)
    
    # elif isinstance(layer, keras.layers.Flatten):
    #   ld = exportFlatten(layer, layers_list)
    #   layers_list.append(ld)

  return layers_list, trainable_layers

