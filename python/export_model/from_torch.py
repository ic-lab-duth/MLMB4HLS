import torch.nn as nn
import copy

# Exports a dictionary from a Conv2D layer
def exportConv2D(layer, layers_list,temp,mlperf=False,dict_temp={}):
  ld = {}
  ld['type'] = 'Conv2D'
  ld['name'] = 'conv_'+str(len([l for l in layers_list if 'Conv2D' in l['type']]))
  if(mlperf):
    ld['R'] = dict_temp['R']
    ld['C'] = dict_temp['C']
    ld['input']=dict_temp['name']
    ld['mlperf']=mlperf
  else:
    ld['R'] = temp['R']
    ld['C'] = temp['C']
  ld['N'] = str(layer.in_channels)
  ld['M'] = str(layer.out_channels)
  ld['K'] = str(layer.kernel_size[0])
  ld['L'] = str(layer.kernel_size[1])
  ld['stride_r'] = str(layer.stride[0])
  ld['stride_c'] = str(layer.stride[1])
  ld['dilation'] = str(layer.dilation[0])
  try:
    ld['groups']=str(layer.groups)
  except:
    ld['groups']=str(1)

  padding = [str(layer.padding[0]), str(layer.padding[0]), str(layer.padding[1]), str(layer.padding[1])]
  if layer.padding_mode != 'zeros':
    print('FIX')
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
def exportMaxPool2D(layer, layers_list,temp):
  ld = {}
  ld['type'] = 'Pool2D'
  ld['name'] = 'max_'+str(len([l for l in layers_list if 'Pool2D' in l['type']]))
  ld['R'] = temp['R']
  ld['C'] = temp['C']
  ld['K'] = str(layer.kernel_size)
  ld['L'] = str(layer.kernel_size)
  ld['M'] = layers_list[-1]['M']
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['Tm'] = '1'
  ld['otype'] = 'signedDataT'
  return ld

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

# def exportReLU(layer, layers_list,temp):
#   ld={}
#   ld['type'] = 'Relu'
#   ld['name'] = 'relu_'+str(len([l for l in layers_list if 'Relu' in l['type']]))
#   ld['R'] = temp['R']
#   ld['C'] = temp['C']
#   ld['M'] = layers_list[-1]['M']
#   ld['Tm'] = '1'
#   ld['Tr'] = '1'
#   ld['Tc'] = '1'
#   ld['otype'] = 'signedDataT'
#   ld['itype'] = 'signedDataT'
#   return ld

def exportFormat(layer, layers_list,temp):
  ld={}
  ld['type'] = 'Format'
  ld['name'] = 'format_'+str(len([l for l in layers_list if 'Format' in l['type']]))
  ld['R'] = temp['R']
  ld['C'] = temp['C']
  ld['M'] = layers_list[-1]['M']
  ld['Tm'] = '1'
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['otype'] = 'signedDataT'
  ld['itype'] = 'signedDataT'
  return ld

def exportReLU6(layer, layers_list,temp):
  ld={}
  ld['type'] = 'Relu6'
  ld['name'] = 'relu6_'+str(len([l for l in layers_list if 'Relu6' in l['type']]))
  ld['R'] = temp['R']
  ld['C'] = temp['C']
  ld['M'] = layers_list[-1]['M']
  ld['Tm'] = '1'
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['otype'] = 'signedDataT'
  ld['itype'] = 'signedDataT'
  return ld

def exportResidual(layer, layers_list,temp,pos_1):
  ld={}
  ld['type'] = 'Residual'
  ld['name'] = 'residual_'+str(len([l for l in layers_list if 'Residual' in l['type']]))
  ld['R'] = temp['R']
  ld['C'] = temp['C']
  ld['M'] = layers_list[-1]['M']
  ld['Tm'] = '1'
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['inp_1']=pos_1
  ld['otype'] = 'signedDataT'
  ld['itype'] = 'signedDataT'
  return ld

def exportAvgPool2D(layer, layers_list,temp,mcunet=False):
  if isinstance(layer, nn.Linear) or isinstance(layer, nn.quantized.Linear):
    ld = {}
    ld['type'] = 'AvgPool2D'
    ld['name'] = 'avg_'+str(len([l for l in layers_list if 'AvgPool2D' in l['type']]))
    ld['R'] = temp['R']
    ld['C'] = temp['C']
    ld['K'] = temp['R']
    ld['L'] = temp['C']
    ld['M'] = layers_list[-1]['M']
    ld['Tr'] = '1'
    ld['Tc'] = '1'
    ld['Tm'] = '1'
    ld['otype'] = 'signedDataT'
    return ld    
  else:
    ld = {}
    ld['type'] = 'AvgPool2D'
    ld['name'] = 'avg_'+str(len([l for l in layers_list if 'AvgPool2D' in l['type']]))
    ld['R'] = temp['R']
    ld['C'] = temp['C']
    try:
      ld['K'] = str(layer.kernel_size)
      ld['L'] = str(layer.kernel_size)
    except:
      stride_0=int(int(temp['R'])/layer.output_size[0])
      stride_1=int(int(temp['C'])/layer.output_size[1])
      ld['K'] = str(int(temp['R'])-(layer.output_size[0]-1)*stride_0)
      ld['L'] = str(int(temp['C'])-(layer.output_size[1]-1)*stride_1)
    ld['M'] = layers_list[-1]['M']
    ld['Tr'] = '1'
    ld['Tc'] = '1'
    ld['Tm'] = '1'
    ld['otype'] = 'signedDataT'
    return ld

def exportSigmoid(layer, layers_list):
  for i in range(len(layers_list)):
    if layers_list[-1-i]['type'] == 'Conv2D' or layers_list[-1-i]['type'] == 'Dense':
      layers_list[-1-i]['activation'] = 'sigmoid'
      break

def exportBatchNorm2d(layer, layers_list,temp):
  ld = {}
  ld['type'] = 'BatchNorm2D'
  ld['name'] = 'batc_'+str(len([l for l in layers_list if l['type']=='BatchNorm2D']))
  ld['R'] = temp['R']
  ld['C'] = temp['C']
  ld['M'] = layers_list[-1]['M']
  ld['Tm'] = '1'
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['otype'] = 'signedDataT'
  ld['itype'] = 'signedDataT'
  return ld

  

def exportFlatten(layer, layers_list,temp):
  ld = {}
  ld['type'] = 'Flatten'
  ld['name'] = 'flat_'+str(len([l for l in layers_list if 'Flatten' in l['type']]))
  ld['R'] = temp['R']
  ld['C'] = temp['C']
  ld['N'] = layers_list[-1]['M']
  ld['Tr'] = '1'
  ld['Tc'] = '1'
  ld['Tm'] = '1'
  ld['Tn'] = '1'
  ld['otype'] = 'signedDataT'
  return ld

# Exports the information from a keras model
def export(model,input,mcunet=False,mlperf=False,list1=[]):
  temp={}
  temp['R']=input['R']
  temp['C']=input['C']
  layers_list = []
  queue=[]
  dict_temp={}
  
  trainable_layers = 0
  flag=False

  for layer in model.modules():

    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.quantized.Conv2d):
      if(mcunet and len(layers_list)>0):
        if(layers_list[-1]['type'] == 'BatchNorm2D' and (layers_list[-1]['name'] in queue)):
          index=queue.index(layers_list[-1]['name'])
          if(index in list1):
            pos_1=queue[index-1]+str("_o")
            ld = exportResidual(layer, layers_list, temp,pos_1)
            layers_list.append(ld)
      if(mlperf and len(layers_list)>0):
        if(layers_list[-1]['type'] == 'BatchNorm2D' and flag==False):
          pos_1=(dict_temp['name']+"_o")
          flag=True
          ld = exportResidual(layer, layers_list, temp,pos_1)
          layers_list.append(ld)
          ld=exportReLU(layer, layers_list,temp)
          layers_list.append(ld)
          dict_temp['name']=ld['name']
          trainable_layers += 1
          ld = exportConv2D(layer, layers_list, temp,mlperf,dict_temp)
          dict_temp['R']=temp['R']
          dict_temp['C']=temp['C']
          temp['R']=str(int((int(temp['R'])+2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1)/layer.stride[0]+1))
          temp['C']=str(int((int(temp['C'])+2*layer.padding[1]-layer.dilation[1]*(layer.kernel_size[1]-1)-1)/layer.stride[1]+1))
        elif(layers_list[-1]['type'] == 'BatchNorm2D'):
          trainable_layers += 1
          ld = exportConv2D(layer, layers_list, temp,mlperf,dict_temp)
          layers_list.append(ld)
          pos_1=queue[-1]+str("_o")
          ld = exportResidual(layer, layers_list, temp,pos_1)
          layers_list.append(ld)
          ld=exportReLU(layer, layers_list,temp)
          dict_temp['R']=temp['R']
          dict_temp['C']=temp['C']
          dict_temp['name']=ld['name']
        elif(layers_list[-1]['type'] == 'Conv2D'):
          pos_1=queue[-1]+str("_o")
          ld = exportResidual(layer, layers_list, temp,pos_1)
          layers_list.append(ld)
          ld=exportReLU(layer, layers_list,temp)
          layers_list.append(ld)
          dict_temp['name']=ld['name']
          trainable_layers += 1
          ld = exportConv2D(layer, layers_list, temp,mlperf,dict_temp)
          dict_temp['R']=temp['R']
          dict_temp['C']=temp['C']
        else:
          trainable_layers += 1
          ld = exportConv2D(layer, layers_list, temp)
          temp['R']=str(int((int(temp['R'])+2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1)/layer.stride[0]+1))
          temp['C']=str(int((int(temp['C'])+2*layer.padding[1]-layer.dilation[1]*(layer.kernel_size[1]-1)-1)/layer.stride[1]+1))
      else:
        trainable_layers += 1
        ld = exportConv2D(layer, layers_list, temp)
        temp['R']=str(int((int(temp['R'])+2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1)/layer.stride[0]+1))
        temp['C']=str(int((int(temp['C'])+2*layer.padding[1]-layer.dilation[1]*(layer.kernel_size[1]-1)-1)/layer.stride[1]+1))
      layers_list.append(ld)


    elif isinstance(layer, nn.ReLU):
      exportReLU(layer, layers_list)
      # if layers_list[-1]['type']=='Dense':
        # ld=exportFormat(layer, layers_list,temp)
        # layers_list.append(ld)
      # ld=exportReLU(layer, layers_list,temp)
      # if ld['name'] =="relu_0" and mlperf:
      #   dict_temp['R']=temp['R']
      #   dict_temp['C']=temp['C']
      #   dict_temp['name']=ld['name']
      # if mlperf:
      #   queue.pop()
      # layers_list.append(ld)

    elif isinstance(layer, nn.ReLU6):
      ld=exportReLU6(layer, layers_list,temp)
      if mcunet:
        queue.pop()
      layers_list.append(ld)

    elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.quantized.BatchNorm2d):
      trainable_layers += 1
      ld = exportBatchNorm2d(layer, layers_list,temp)
      temp['R']=temp['R']
      temp['C']=temp['C']
      queue.append(ld['name'])
      layers_list.append(ld)

    elif isinstance(layer, nn.Sigmoid):
      exportSigmoid(layer, layers_list)

    elif isinstance(layer, nn.Linear) or isinstance(layer, nn.quantized.Linear):
      if mcunet:
        ld = exportAvgPool2D(layer, layers_list,temp,mcunet)
        layers_list.append(ld)
        temp['R']=str(1)
        temp['C']=str(1)
      if len(layers_list)>0:
        if layers_list[-1]['type']!='Flatten' and layers_list[-1]['type']!='Dense':
          ld = exportFlatten(layer, layers_list,temp)
          layers_list.append(ld)
      trainable_layers += 1
      ld = exportLinear(layer, layers_list)
      layers_list.append(ld)

    elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.quantized.MaxPool2d):
      ld = exportMaxPool2D(layer, layers_list,temp)
      temp['R']=str(int((int(temp['R'])+2*layer.padding-layer.dilation*(layer.kernel_size-1)-1)/layer.stride+1))
      temp['C']=str(int((int(temp['C'])+2*layer.padding-layer.dilation*(layer.kernel_size-1)-1)/layer.stride+1))
      layers_list.append(ld)
    
    elif isinstance(layer, nn.Flatten):
      ld = exportFlatten(layer, layers_list,temp)
      layers_list.append(ld)

    elif isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d):
      ld = exportAvgPool2D(layer, layers_list,temp)
      temp['R']=str(int(int(temp['R'])/int(ld['K'])))
      temp['C']=str(int(int(temp['C'])/int(ld['L'])))
      layers_list.append(ld)

  return layers_list, trainable_layers

