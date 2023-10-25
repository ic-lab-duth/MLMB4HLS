import keras

# Exports a dictionary from a Conv2D layer
def exportConv2D(layer, conf, layers_list):
  ld = {}
  ld['type'] = 'Conv2D'
  ld['name'] = layer.name
  ld['R'] = str(layer.input_shape[1])
  ld['C'] = str(layer.input_shape[2])
  ld['N'] = str(layer.input_shape[3])
  ld['M'] = str(layer.output_shape[3])
  ld['K'] = str(conf['kernel_size'][0])
  ld['L'] = str(conf['kernel_size'][1])
  ld['stride'] = str(conf['strides'][0])
  ld['dilation'] = str(conf['dilation_rate'][0])

  padding = ['0', '0', '0', '0']
  if conf['padding'] == 'same':
    for i in range(4):
      padding[i] = str(int((conf['kernel_size'][0]-1)/2))
  ld['padding'] = padding

  if conf['activation'] == 'relu':
    ACT = 'relu'  
    otype = 'unsignedDataT'
  else:
    ACT = 'linear'
    otype = 'signedDataT'

  ld['otype'] = otype
  ld['itype'] = 'unsignedDataT' if len(layers_list)==0 else layers_list[-1]['otype']
  ld['activation'] = ACT
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['Tm'] = '1'
  ld['Tn'] = '1'
  return ld

# Exports a dictionary from a MaxPooling2D layer
def exportMaxPool2D(layer, layers_list):
  ld = {}
  ld['type'] = 'Pool2D'
  ld['name'] = layer.name
  ld['R'] = str(layer.input_shape[1])
  ld['C'] = str(layer.input_shape[2])
  ld['M'] = str(layer.input_shape[3])
  ld['K'] = str(layer.pool_size[0])
  ld['L'] = str(layer.pool_size[0])
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['Tm'] = '1'
  ld['otype'] = layers_list[-1]['otype']
  return ld

# Exports a dictionary from a Dense layer
def exportDense(layer, conf, layers_list):
  ld = {}
  ld['type'] = 'Dense'
  ld['name'] = layer.name
  ld['M'] = str(layer.output_shape[1])
  ld['N'] = str(layer.input_shape[1])

  if conf['activation'] == 'relu':
    ACT = 'relu'  
    dtype = 'unsignedDataT'
  elif  conf['activation'] == 'softmax':
    ACT = 'softmax'
    dtype = 'signedDataT'
  else:
    ACT = 'linear'
    dtype = 'signedDataT'

  ld['otype'] = dtype
  ld['itype'] = 'unsignedDataT' if len(layers_list)==0 else layers_list[-1]['otype']
  ld['activation'] = ACT
  ld['Tn'] = '1'
  ld['Tm'] = '1'
  return ld

def exportFlatten(layer, layers_list):
  ld = {}
  ld['type'] = 'Flatten'
  ld['name'] = layer.name
  if layers_list[-1]['type'] == 'Pool2D':
    ld['R'] = str(int(int(layers_list[-1]['R'])/int(layers_list[-1]['K'])))
    ld['C'] = str(int(int(layers_list[-1]['C'])/int(layers_list[-1]['L'])))
  else:
    ld['R'] = layers_list[-1]['R']
    ld['C'] = layers_list[-1]['C']

  ld['N'] = layers_list[-1]['M']
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['Tm'] = '1'
  ld['Tn'] = '1'
  ld['otype'] = layers_list[-1]['otype']
  return ld

# Exports the information from a keras model
def export(model):
  layers_list = []
  trainable_layers = 0

  for layer in model.layers:
    conf = layer.get_config()

    if isinstance(layer, keras.layers.Conv2D):
      trainable_layers += 1
      ld = exportConv2D(layer, conf, layers_list)
      layers_list.append(ld)

    elif isinstance(layer, keras.layers.MaxPooling2D):
      ld = exportMaxPool2D(layer, layers_list)
      layers_list.append(ld)
      
    elif isinstance(layer, keras.layers.ReLU):
      # TODO: 
      print('relu')

    elif isinstance(layer, keras.layers.Dense):
      trainable_layers += 1
      ld = exportDense(layer, conf, layers_list)
      layers_list.append(ld)
    
    elif isinstance(layer, keras.layers.Flatten):
      ld = exportFlatten(layer, layers_list)
      layers_list.append(ld)

  return layers_list, trainable_layers

