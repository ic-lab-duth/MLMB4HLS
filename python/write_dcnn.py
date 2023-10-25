# Write a Conv2D layer 
def defineConv2D_Obj(layer):
  line = 'Conv2D<'
  line += layer['itype'] + ', '
  line += layer['otype'] + ', '
  line += layer['R'] + ', '
  line += layer['C'] + ', '
  line += layer['N'] + ', '
  line += layer['M'] + ', '
  line += layer['K'] + ', '
  line += layer['L'] + ', '
  line += layer['Tm'] + ', '
  line += layer['Tn'] + ', '
  line += layer['Tr'] + ', '
  line += layer['Tc'] + ', '
  line += layer['padding'][0] + ', '
  line += layer['padding'][1] + ', '
  line += layer['padding'][2] + ', '
  line += layer['padding'][3] + ', '
  line += layer['activation'] + '>  '
  line += layer['name'] +';'
  return line

# Writes a Pool2D layer (Max) 
def definePool2D_Obj(layer):
  line = 'Pool2D<'
  line += layer['otype'] + ', '
  line += layer['R']  + ', '
  line += layer['C']  + ', '
  line += layer['M']  + ', '
  line += layer['K']  + ', '
  line += layer['L']  + ', '
  line += layer['Tm'] + ', '
  line += layer['Tr'] + ', '
  line += layer['Tc'] + ', '
  line += 'Max> '
  line += layer['name'] + ';'
  return line

#Writed a Desnse layer
def defineDense_Obj(layer):
  line = 'Dense<'
  line += layer['itype'] + ', '
  line += layer['otype'] + ', '
  line += layer['N']  + ', '
  line += layer['M']  + ', '
  line += layer['Tm'] + ', '
  line += layer['Tn'] + ', '
  line += layer['activation'] + '>  '
  line += layer['name'] + ';'
  return line

#Writed a Flatten layer
def defineFlatten_Obj(layer):
  line = 'Flatten<'
  line += layer['otype'] + ', '
  line += layer['R']  + ', '
  line += layer['C']  + ', '
  line += layer['N'] + ', '
  line += layer['Tn'] + ', '
  line += layer['Tm'] + ', '
  line += layer['Tr'] + ', '
  line += layer['Tc'] + '>  '
  line += layer['name'] + ';'
  return line

#Writed a BatchNorm2D layer
def defineBatchNorm2D_Obj(layer):
  line = 'BatchNormalization<'
  line += layer['itype'] + ', '
  line += layer['otype'] + ', '
  line += layer['R']  + ', '
  line += layer['C']  + ', '
  line += layer['M'] + ', '
  line += layer['Tm'] + ', '
  line += layer['Tr'] + ', '
  line += layer['Tc'] + '>  '
  line += layer['name'] + ';'
  return line

# Creates a new line
def new_line(f):
  f.write('\n')

# Writes a new line with a defined indentation
def write_line(f, indent, line):
  indentation_list = ['', 
                      '  ', 
                      '    ', 
                      '      ',
                      '        ',
                      '          ',
                      '            ',
                      '              ']
  f.write(indentation_list[indent] + line + '\n')


# Writes the header file includes of the file
def write_includes(f, hList):
  for h in hList:
    write_line(f, 0 ,'#include ' + h)
  new_line(f)

# Writes the typedefs of the class
def write_typedefs(f, lList, interface):
  write_line(f,1,'typedef unsignedDataT     dtype_I;')
  write_line(f,1,'typedef signedDataT       dtype_O;')
  write_line(f,1,'typedef weightT           wtype;')
  write_line(f,1,'typedef biasT             btype;')
  new_line(f)
  if interface == 'VECTOR':
    write_line(f,1,'typedef ndmatrix::Mat3d<dtype_I, '+lList[0]['R']+', '+lList[0]['C']+', '+lList[0]['N']+'> chanI;')
    write_line(f,1,'typedef ndmatrix::Mat1d<dtype_I, '+lList[-1]['M']+'> chanO;')
    
    i = 0
    for layer in lList:
      if layer['type'] == 'Conv2D':
        write_line(f,1,'typedef ndmatrix::Mat4d<wtype,'+layer['M']+','+layer['N']+','+layer['K']+','+layer['L']+'> chanW_'+str(i)+';')
        write_line(f,1,'typedef ndmatrix::Mat1d<btype, '+layer['M']+'> chanB_'+str(i)+';')
        i += 1
      elif layer['type'] == 'Dense':
        write_line(f,1,'typedef ndmatrix::Mat2d<wtype,'+layer['N']+','+layer['M']+'> chanW_'+str(i)+';')
        write_line(f,1,'typedef ndmatrix::Mat1d<btype, '+layer['M']+'> chanB_'+str(i)+';')
        i += 1
  else:
    write_line(f,1,'typedef compactDataT<dtype_I, '+lList[0]['Tn']+'> unrolled_dti;')
    write_line(f,1,'typedef compactDataT<dtype_O, '+lList[-1]['Tm']+'> unrolled_dto;')
    new_line(f)
    write_line(f,1,'typedef ac_channel<unrolled_dti> chanI;')
    write_line(f,1,'typedef ac_channel<unrolled_dto> chanO;')
    write_line(f,1,'typedef ac_channel<bool>         chanL;')
    write_line(f,1,'typedef ac_channel<wtype>        chanW;')
    write_line(f,1,'typedef ac_channel<btype>        chanB;')
  
  new_line(f)

# Writes the insantiation of the layer objects
def write_layers(f, lList):
  identation = 1
  for l in lList:
    if l['type'] == 'Conv2D':
      line = defineConv2D_Obj(l)
    elif l['type'] == 'Pool2D':
      line = definePool2D_Obj(l)
    elif l['type'] == 'Dense':
      line = defineDense_Obj(l)
    elif l['type'] == 'Flatten':
      line = defineFlatten_Obj(l)
    elif l['type'] == 'BatchNorm2D':
      line = defineBatchNorm2D_Obj(l)
    else:
      line = ''
    write_line(f, identation, line)
  new_line(f)

# Writes the channels need for the interconnection
def write_channels(f, lList, interface):
  for l in lList:
    if l != lList[-1]:
      dtype = l['otype']
      if interface == 'VECTOR':
        if l['type'] == 'Dense':
          line = 'ndmatrix::Mat1d<'+dtype+','+l['M']+'> '
        elif l['type'] == 'Flatten':
          print(l['R'],l['C'],l['N'])
          line = 'ndmatrix::Mat1d<'+dtype+','+str(int(l['R'])*int(l['C'])*int(l['N']))+'> '
        elif l['type'] == 'Conv2D':
          line = 'ndmatrix::Mat3d<'+dtype+','+l['R']+','+l['C']+','+l['M']+'> '
        elif l['type'] == 'Pool2D':
          line = 'ndmatrix::Mat3d<'+dtype+','+str(int(int(l['R'])/int(l['K'])))+','+str(int(int(l['C'])/int(l['L'])))+','+l['M']+'> '
          
      else:
        line = 'ac_channel<compactDataT<'+dtype+', ' + l['Tm'] + '>> '
      line += l['name']+'_o;'
      write_line(f, 1, line)
  new_line(f)

# Writes the class Constructor/Destructor
def write_constructors(f, className):
  write_line(f, 1, className + '() {};')
  write_line(f, 1, '~' + className + '() {};')
  new_line(f)

def write_interconnection(f, lList, interface):
  iter = 0
  input = 'inp'
  for l in lList:
    if l['type'] == 'Conv2D':
      line = l['name']+'.run('
      if interface == 'VECTOR':
        line += 'w' +str(iter)+ ', '
        line += 'b' +str(iter)+ ', '
        line += input + ', '
      else:
        line += 'l[' +str(iter)+ '], '
        line += 'w[' +str(iter)+ '], '
        line += 'b[' +str(iter)+ '], '
        line += input + ', '

      if l != lList[-1]:
        line += l['name'] +'_o);'
      else:
        line += 'out);'
      
      iter = iter + 1
      input = l['name']+'_o'
      write_line(f, 2, line)

    elif l['type'] == 'Pool2D':
      line = l['name']+'.run('
      line += input + ', '
      if l != lList[-1]:
        line += l['name'] +'_o);'
      else:
        line += 'out);'
      
      input = l['name']+'_o'
      write_line(f, 2, line)
    
    elif l['type'] == 'Dense':
      line = l['name']+'.run('
      if interface == 'VECTOR':
        line += 'w' +str(iter)+ ', '
        line += 'b' +str(iter)+ ', '
        line += input + ', '
      else:
        line += 'l[' +str(iter)+ '], '
        line += 'w[' +str(iter)+ '], '
        line += 'b[' +str(iter)+ '], '
        line += input + ', '
      if l != lList[-1]:
        line += l['name'] +'_o);'
      else:
        line += 'out);'
      
      iter = iter + 1
      input = l['name']+'_o'
      write_line(f, 2, line)
    
    elif l['type'] == 'Flatten':
      line = l['name']+'.run('
      line += input + ', '
      if l != lList[-1]:
        line += l['name'] +'_o);'
      else:
        line += 'out);'
      
      input = l['name']+'_o'
      write_line(f, 2, line)

    elif l['type'] == 'BatchNorm2D':
      line = l['name']+'.run('
      if interface == 'VECTOR':
        line += 'w' +str(iter)+ ', '
        line += 'b' +str(iter)+ ', '
        line += input + ', '
      else:
        line += 'l[' +str(iter)+ '], '
        line += 'w[' +str(iter)+ '], '
        line += 'b[' +str(iter)+ '], '
        line += input + ', '

      if l != lList[-1]:
        line += l['name'] +'_o);'
      else:
        line += 'out);'
      
      iter = iter + 1
      input = l['name']+'_o'
      write_line(f, 2, line)

def write_top_func(f, trainable_layers, interface):
  write_line(f,1, '#pragma hls_design interface')
  if interface == 'VECTOR':
    line = 'void predict('
    for i in range(trainable_layers):
      line += 'chanW_' + str(i) + '&w' + str(i) + ', '
      line += 'chanB_' + str(i) + '&b' + str(i) + ', '
      write_line(f,1,line)
      line = ''

    line = 'chanI &inp, chanO &out) {'
    write_line(f,1,line)
  
  else:
    line = 'void predict(chanL l['+str(trainable_layers)+'], '
    line += 'chanW w['+str(trainable_layers)+'], '
    line += 'chanB b['+str(trainable_layers)+'], '
    line += 'chanI &inp, chanO &out) {'
    write_line(f,1,line)

# Writes the content of the private block of the class
def write_private(f, lList, interface):
  write_line(f,0,'private:')
  new_line(f)

  write_typedefs(f, lList, interface)
  write_layers(f, lList)
  write_channels(f, lList, interface)

# Writes the content of the public block of the class
def write_public(f, className, interface, lList, trainable_layers):
  write_line(f,0,'public:')
  write_constructors(f, className)

  write_top_func(f, trainable_layers, interface)
  new_line(f)
  write_interconnection(f, lList, interface)
  new_line(f)
  write_line(f,1,'}; // (function) predict')

# Write the whole class 
def write_class(f, className, interface, lList, trainable_layers):
  write_line(f,0, '#pragma hls_design')
  write_line(f,0,'class '+className+' {')
  write_private(f, lList, interface)
  write_public(f, className, interface, lList, trainable_layers)
  write_line(f,0,'}; // (class) '+className)
  new_line(f)
  new_line(f)


def write_testbench(parameters):
  f = open(parameters['filename_tb'], 'w') 

  lList = parameters['layers']

  write_line(f,0,'#include "dcnn_'+parameters['name']+'.h"')
  write_line(f,0,'#include "helpers.h"')
  new_line(f)

  write_line(f,0,'int main(int argc, char* argv[]) {')
  new_line(f)

  input_file = '"'+parameters['workspace']+'/examples/'+parameters['name']+'/images/dog.txt"'
  output_file = '"'+parameters['workspace']+'/examples/'+parameters['name']+'/output.txt"'

  if parameters['interface'] == 'VECTOR':
    if lList[0]['type'] == 'Conv2D':
      write_line(f,1, 'ndmatrix::Mat3d<dcnn::unsignedDataT,'+lList[0]['R']+', '+lList[0]['C']+', '+lList[0]['N']+'> inp;')
      write_line(f,1, 'read_3d_ndarray_from_txt<dcnn::unsignedDataT,'+lList[0]['R']+', '+lList[0]['C']+', '+lList[0]['N']+'>(inp, '+input_file+');')
    elif lList[0]['type'] == 'Dense':
      write_line(f,1, 'ndmatrix::Mat1d<dcnn::unsignedDataT,'+lList[0]['N']+'> inp;')
      write_line(f,1, 'read_1d_ndarray_from_txt<dcnn::unsignedDataT,'+lList[0]['N']+'>(inp, '+input_file+');')
    # elif lList[0]['type'] == 'BatchNorm2D':
    #   write_line(f,1, 'ndmatrix::Mat1d<dcnn::unsignedDataT,'+lList[0]['N']+'> inp;')
    #   write_line(f,1, 'read_1d_ndarray_from_txt<dcnn::unsignedDataT,'+lList[0]['N']+'>(inp, '+input_file+');')

                 
    new_line(f)


    i = 0
    for l in lList:
      line = ''
      if l['type'] == 'Conv2D':
        line += 'ndmatrix::Mat4d<dcnn::weightT, '
        line += l['M']+', '
        line += l['N']+', '
        line += l['K']+', '
        line += l['L']+'> w'+str(i)+';'
        write_line(f,1, line)
        line = 'read_4d_ndarray_from_txt<dcnn::weightT, '
        line += l['M']+', '
        line += l['N']+', '
        line += l['K']+', '
        line += l['L']+'>(w'+str(i)+', '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/weights/'+l['name']+'_w.txt");'
        write_line(f,1, line)
        line = 'ndmatrix::Mat1d<dcnn::biasT, '
        line += l['M']+'> b'+str(i)+';'
        write_line(f,1, line)
        line = 'read_1d_ndarray_from_txt<dcnn::biasT, '
        line += l['M']+'>(b'+str(i)+', '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/biases/'+l['name']+'_b.txt");'
        write_line(f,1, line)
        i += 1
      elif l['type'] == 'Dense':
        line += 'ndmatrix::Mat2d<dcnn::weightT, '
        line += l['N']+', '
        line += l['M']+'> w'+str(i)+';'
        write_line(f,1, line)
        line = 'read_2d_ndarray_from_txt<dcnn::weightT, '
        line += l['N']+', '
        line += l['M']+'>(w'+str(i)+', '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/weights/'+l['name']+'_w.txt");'
        write_line(f,1, line)
        line = 'ndmatrix::Mat1d<dcnn::biasT, '
        line += l['M']+'> b'+str(i)+';'
        write_line(f,1, line)
        line = 'read_1d_ndarray_from_txt<dcnn::biasT, '
        line += l['M']+'>(b'+str(i)+', '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/biases/'+l['name']+'_b.txt");'
        write_line(f,1, line)
        i += 1
      new_line(f)
    
    write_line(f,1, 'ndmatrix::Mat1d<dcnn::unsignedDataT, '+lList[-1]['M']+'> out;')
    new_line(f)

    write_line(f,1, 'dcnn::cpp::'+parameters['name']+' model;')
    new_line(f)

    write_line(f,1, 'model.predict( ')
    
    for j in range(i):
      write_line(f,2, 'w'+str(j)+', b'+str(j)+', ')
    
    write_line(f,2, 'inp, out);')
    new_line(f)

    write_line(f,1, 'write_1d_ndarray_to_txt<float, '+lList[-1]['M']+'>(out, '+output_file+');')
    new_line(f)

  else:
    write_line(f,1, 'ac_channel<dcnn::unsignedDataT> tmp_inp;')
    write_line(f,1, 'ac_channel<dcnn::compactDataT<dcnn::unsignedDataT,'+lList[0]['Tn']+'>> inp;')
    if lList[0]['type'] == 'Conv2D':
      write_line(f,1, 'write_image_to_channel<dcnn::unsignedDataT,'+lList[0]['R']+','+lList[0]['C']+','+lList[0]['N']+'>(tmp_inp, '+input_file+');')
    elif lList[0]['type'] == 'Dense':
      write_line(f,1, 'write_bias_to_channel<dcnn::unsignedDataT,'+lList[0]['N']+'>(inp, '+input_file+');')
    new_line(f)

    write_line(f,1, 'while(tmp_inp.available(1)) {')
    write_line(f,2, 'dcnn::compactDataT<dcnn::unsignedDataT,'+lList[0]['Tn']+'> pack;')
    write_line(f,2, 'for (int i = 0; i < '+lList[0]['Tn']+'; i++) {')
    write_line(f,3, 'if (tmp_inp.available(1)) {')
    write_line(f,4, 'pack[i] = tmp_inp.read();')
    write_line(f,3, '}')
    write_line(f,2, '}')
    write_line(f,2, 'inp.write(pack);')
    write_line(f,1, '}')
    new_line(f)

    write_line(f,1, 'ac_channel<dcnn::weightT> w['+str(parameters['trainable'])+'];')
    write_line(f,1, 'ac_channel<dcnn::biasT>   b['+str(parameters['trainable'])+'];')
    write_line(f,1, 'ac_channel<bool>          l['+str(parameters['trainable'])+'];')
    new_line(f)

    i = 0
    for l in lList:
      if l['type'] == 'Conv2D':
        line = 'write_weights_to_channel<dcnn::weightT, '
        line += l['M']+', '
        line += l['N']+', '
        line += l['K']+', '
        line += l['L']+'>(w['+str(i)+'], '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/weights/'+l['name']+'_w.txt");'
        write_line(f,1, line)
        line = 'write_bias_to_channel<dcnn::biasT, '+l['M']+'>(b['+str(i)+'], '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/biases/'+l['name']+'_b.txt");'
        write_line(f,1, line)
        i += 1
      elif l['type'] == 'Dense':
        line = 'write_dense_weights_to_channel<dcnn::weightT, '
        line += l['N']+', '
        line += l['M']+'>(w['+str(i)+'], '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/weights/'+l['name']+'_w.txt");'
        write_line(f,1, line)
        line = 'write_bias_to_channel<dcnn::biasT, '+l['M']+'>(b['+str(i)+'], '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/biases/'+l['name']+'_b.txt");'
        write_line(f,1, line)
        i += 1
      elif l['type'] == 'BatchNorm2D':
        line = 'write_weights_to_channel<dcnn::weightT, '
        line += l['M']+', '
        line += l['N']+', '
        line += l['K']+', '
        line += l['L']+'>(w['+str(i)+'], '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/weights/'+l['name']+'_w.txt");'
        write_line(f,1, line)
        line = 'write_bias_to_channel<dcnn::biasT, '+l['M']+'>(b['+str(i)+'], '
        line += '"'+parameters['workspace']+'/examples/'+parameters['name']+'/parameters/biases/'+l['name']+'_b.txt");'
        write_line(f,1, line)
        i += 1
      new_line(f)

    write_line(f,1, 'ac_channel<dcnn::compactDataT<dcnn::unsignedDataT,'+lList[-1]['Tm']+'>> out;')
    new_line(f)

    write_line(f,1, 'dcnn::hls::'+parameters['name']+' model;')
    new_line(f)

    write_line(f, 1, 'for (int i = 0; i < '+str(parameters['trainable'])+'; i++) {')
    write_line(f,2, 'l[i].write(true);')
    write_line(f,1,'}')
    new_line(f)
    write_line(f,1, 'model.predict(l, w, b, inp, out);')   
    new_line(f)

    write_line(f, 1, 'for (int i = 0; i < '+str(parameters['trainable'])+'; i++) {')
    write_line(f,2, 'l[i].write(false);')
    write_line(f,1,'}')
    new_line(f)
    write_line(f,1, 'model.predict(l, w, b, inp, out);') 
    new_line(f) 

    write_line(f,1, 'ac_channel<dcnn::unsignedDataT> dout;')
    write_line(f,1, 'while(out.available(1)) {')
    write_line(f,2, 'dcnn::compactDataT<dcnn::unsignedDataT,'+lList[-1]['Tm']+'> pack;')
    write_line(f,2, 'pack = out.read();')
    write_line(f,2, 'for (int i = 0; i < '+lList[-1]['Tm']+'; i++) {')
    write_line(f,3, 'dout.write(pack[i]);')
    write_line(f,2, '}')
    write_line(f,1, '}')
    new_line(f)
    
    write_line(f,1, 'write_1d_channel_data_to_txt<dcnn::unsignedDataT, '+lList[-1]['M']+'>(dout, '+output_file+');')
    new_line(f)


  write_line(f,1, '// Extra code here')
  new_line(f)
  new_line(f)
  
  write_line(f,1, 'return 0;')
  write_line(f,0, '}')


  f.close()


# Write a cpp header file for a keras model
def defineModel(parameters):
  f = open(parameters['filename'], 'w') 

  write_line(f,0,'#ifndef '+parameters['headerDef'])
  write_line(f,0,'#define '+parameters['headerDef'])
  new_line(f)

  write_includes(f, parameters['includes'])

  if parameters['in_namespace']:
    write_line(f,0,'namespace '+parameters['namespace']+' {')
    write_line(f,0,'namespace '+parameters['namespace_2']+' {')
    new_line(f)

  write_class(f, parameters['name'], parameters['interface'], parameters['layers'], parameters['trainable'])

  if parameters['in_namespace']:
    write_line(f,0,'}; // (namespace) '+parameters['namespace_2'])
    write_line(f,0,'}; // (namespace) '+parameters['namespace'])
    new_line(f)

  write_line(f,0,'#endif')
  new_line(f)

  f.close()

  write_testbench(parameters)