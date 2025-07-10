
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 32, 32, 32, 3).astype(np.float32)
    ksize1 = [1, 2, 2, 2, 1]
    strides1 = [1, 2, 2, 2, 1]
    padding1 = 'VALID'
    data_format1 = 'NDHWC'
    name1 = 'max_pool_1'
    input_dict1 = {'input': tf.convert_to_tensor(input1).numpy(), 'ksize': ksize1, 'strides': strides1, 'padding': padding1, 'data_format': data_format1, 'name': name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(1, 16, 16, 16, 5).astype(np.float32)
    ksize2 = [1, 2, 2, 2, 1] #modified ksize
    strides2 = [1, 2, 2, 2, 1] #modified strides
    padding2 = 'SAME'
    data_format2 = 'NDHWC'
    name2 = 'max_pool_2'
    input_dict2 = {'input': tf.convert_to_tensor(input2).numpy(), 'ksize': ksize2, 'strides': strides2, 'padding': padding2, 'data_format': data_format2, 'name': name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(2, 8, 8, 8, 10).astype(np.float32)
    ksize3 = [1, 1, 2, 2, 1] #modified ksize
    strides3 = [1, 1, 2, 2, 1] #modified strides
    padding3 = 'VALID'
    data_format3 = 'NDHWC'
    name3 = 'max_pool_3'
    input_dict3 = {'input': tf.convert_to_tensor(input3).numpy(), 'ksize': ksize3, 'strides': strides3, 'padding': padding3, 'data_format': data_format3, 'name': name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
    ksize4 = [1, 1, 3, 3, 1] #modified ksize
    strides4 = [1, 1, 2, 2, 1] #modified strides
    padding4 = 'SAME'
    data_format4 = 'NDHWC'
    name4 = 'max_pool_4'
    input_dict4 = {'input': tf.convert_to_tensor(input4).numpy(), 'ksize': ksize4, 'strides': strides4, 'padding': padding4, 'data_format': data_format4, 'name': name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

     # Input 5
    input5 = np.random.rand(1, 3, 3, 3, 1).astype(np.float32)
    ksize5 = [1, 1, 2, 2, 1] #modified ksize
    strides5 = [1, 1, 1, 1, 1] #modified strides
    padding5 = 'VALID'
    data_format5 = 'NDHWC'
    name5 = 'max_pool_5'
    input_dict5 = {'input': tf.convert_to_tensor(input5).numpy(), 'ksize': ksize5, 'strides': strides5, 'padding': padding5, 'data_format': data_format5, 'name': name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.rand(1, 32, 32, 32, 3).astype(np.float32)
    ksize6 = [1, 2, 2, 2, 1]
    strides6 = [1, 2, 2, 2, 1]
    padding6 = 'VALID'
    data_format6 = 'NDHWC' # Changed data_format back to NDHWC to avoid CPU error
    name6 = 'max_pool_6'
    input_dict6 = {'input': tf.convert_to_tensor(input6).numpy(), 'ksize': ksize6, 'strides': strides6, 'padding': padding6, 'data_format': data_format6, 'name': name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.rand(1, 3, 16, 16, 16).astype(np.float32)
    ksize7 = [1, 1, 4, 4, 1] #modified ksize
    strides7 = [1, 1, 4, 4, 1] #modified strides
    padding7 = 'SAME'
    data_format7 = 'NDHWC' # Changed data_format back to NDHWC to avoid CPU error
    name7 = 'max_pool_7'
    input_dict7 = {'input': tf.convert_to_tensor(input7).numpy(), 'ksize': ksize7, 'strides': strides7, 'padding': padding7, 'data_format': data_format7, 'name': name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(2, 10, 8, 8, 8).astype(np.float32)
    ksize8 = [1, 1, 2, 2, 1] #modified ksize
    strides8 = [1, 1, 1, 1, 1] #modified strides
    padding8 = 'VALID'
    data_format8 = 'NDHWC' # Changed data_format back to NDHWC to avoid CPU error
    name8 = 'max_pool_8'
    input_dict8 = {'input': tf.convert_to_tensor(input8).numpy(), 'ksize': ksize8, 'strides': strides8, 'padding': padding8, 'data_format': data_format8, 'name': name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.random.rand(2, 3, 5, 5, 5).astype(np.float32)
    ksize9 = [1, 1, 3, 3, 1] #modified ksize
    strides9 = [1, 1, 2, 2, 1] #modified strides
    padding9 = 'SAME'
    data_format9 = 'NDHWC' # Changed data_format back to NDHWC to avoid CPU error
    name9 = 'max_pool_9'
    input_dict9 = {'input': tf.convert_to_tensor(input9).numpy(), 'ksize': ksize9, 'strides': strides9, 'padding': padding9, 'data_format': data_format9, 'name': name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

     # Input 10
    input10 = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
    ksize10 = [1, 1, 2, 2, 1] #modified ksize
    strides10 = [1, 1, 1, 1, 1] #modified strides
    padding10 = 'VALID'
    data_format10 = 'NDHWC' # Changed data_format back to NDHWC to avoid CPU error
    name10 = 'max_pool_10'
    input_dict10 = {'input': tf.convert_to_tensor(input10).numpy(), 'ksize': ksize10, 'strides': strides10, 'padding': padding10, 'data_format': data_format10, 'name': name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.max_pool3d"] = tf_nn_max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.max_pool3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool3d'.")

check_valid('tf.nn.max_pool3d', generated_inputs['tf.nn.max_pool3d'], lib="tf", suffix=0)
