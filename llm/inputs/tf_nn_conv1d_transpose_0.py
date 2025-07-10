
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_transpose_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 3).astype(np.float32)
    filters1 = np.random.rand(3, 2, 3).astype(np.float32)
    output_shape1 = np.array([1, 7, 2]).astype(np.int32)
    strides1 = 1
    padding1 = 'SAME'
    data_format1 = 'NWC'
    dilations1 = 1
    name1 = 'transpose_conv1d_1'

    input_dict1 = {
        'input': tf.convert_to_tensor(input1),
        'filters': tf.convert_to_tensor(filters1),
        'output_shape': tf.convert_to_tensor(output_shape1),
        'strides': strides1,
        'padding': padding1,
        'data_format': data_format1,
        'dilations': dilations1,
        'name': name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(2, 10, 4).astype(np.float32)
    filters2 = np.random.rand(5, 6, 4).astype(np.float32)
    output_shape2 = np.array([2, 14, 6]).astype(np.int32)
    strides2 = 2
    padding2 = 'VALID'
    data_format2 = 'NWC'
    dilations2 = 1
    name2 = 'transpose_conv1d_2'

    input_dict2 = {
        'input': tf.convert_to_tensor(input2),
        'filters': tf.convert_to_tensor(filters2),
        'output_shape': tf.convert_to_tensor(output_shape2),
        'strides': strides2,
        'padding': padding2,
        'data_format': data_format2,
        'dilations': dilations2,
        'name': name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(1, 7, 5).astype(np.float32)
    filters3 = np.random.rand(2, 3, 5).astype(np.float32)
    output_shape3 = np.array([1, 8, 3]).astype(np.int32)
    strides3 = 1
    padding3 = 'VALID'
    data_format3 = 'NWC'
    dilations3 = 2
    name3 = 'transpose_conv1d_3'

    input_dict3 = {
        'input': tf.convert_to_tensor(input3),
        'filters': tf.convert_to_tensor(filters3),
        'output_shape': tf.convert_to_tensor(output_shape3),
        'strides': strides3,
        'padding': padding3,
        'data_format': data_format3,
        'dilations': dilations3,
        'name': name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(3, 12, 2).astype(np.float32)
    filters4 = np.random.rand(4, 4, 2).astype(np.float32)
    output_shape4 = np.array([3, 15, 4]).astype(np.int32)
    strides4 = 1
    padding4 = 'SAME'
    data_format4 = 'NWC'
    dilations4 = 3
    name4 = 'transpose_conv1d_4'

    input_dict4 = {
        'input': tf.convert_to_tensor(input4),
        'filters': tf.convert_to_tensor(filters4),
        'output_shape': tf.convert_to_tensor(output_shape4),
        'strides': strides4,
        'padding': padding4,
        'data_format': data_format4,
        'dilations': dilations4,
        'name': name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(1, 8, 3).astype(np.float32)
    filters5 = np.random.rand(3, 2, 3).astype(np.float32)
    output_shape5 = np.array([1, 10, 2]).astype(np.int32)
    strides5 = 1
    padding5 = 'SAME'
    data_format5 = 'NCW'
    dilations5 = 1
    name5 = 'transpose_conv1d_5'

    input_dict5 = {
        'input': tf.convert_to_tensor(input5),
        'filters': tf.convert_to_tensor(filters5),
        'output_shape': tf.convert_to_tensor(output_shape5),
        'strides': strides5,
        'padding': padding5,
        'data_format': data_format5,
        'dilations': dilations5,
        'name': name5
    }
    input_dict5['input'] = tf.transpose(input_dict5['input'], perm=[0, 2, 1])
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.rand(1, 5, 3).astype(np.float32)
    filters6 = np.random.rand(3, 2, 3).astype(np.float32)
    output_shape6 = np.array([1, 7, 2]).astype(np.int32)
    strides6 = 1
    padding6 = 'SAME'
    data_format6 = 'NCW'
    dilations6 = 1
    name6 = 'transpose_conv1d_6'

    input_dict6 = {
        'input': tf.convert_to_tensor(input6),
        'filters': tf.convert_to_tensor(filters6),
        'output_shape': tf.convert_to_tensor(output_shape6),
        'strides': strides6,
        'padding': padding6,
        'data_format': data_format6,
        'dilations': dilations6,
        'name': name6
    }
    input_dict6['input'] = tf.transpose(input_dict6['input'], perm=[0, 2, 1])
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7
    input7 = np.random.rand(2, 10, 4).astype(np.float32)
    filters7 = np.random.rand(5, 6, 4).astype(np.float32)
    output_shape7 = np.array([2, 14, 6]).astype(np.int32)
    strides7 = 2
    padding7 = 'VALID'
    data_format7 = 'NCW'
    dilations7 = 1
    name7 = 'transpose_conv1d_7'

    input_dict7 = {
        'input': tf.convert_to_tensor(input7),
        'filters': tf.convert_to_tensor(filters7),
        'output_shape': tf.convert_to_tensor(output_shape7),
        'strides': strides7,
        'padding': padding7,
        'data_format': data_format7,
        'dilations': dilations7,
        'name': name7
    }
    input_dict7['input'] = tf.transpose(input_dict7['input'], perm=[0, 2, 1])
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(1, 7, 5).astype(np.float32)
    filters8 = np.random.rand(2, 3, 5).astype(np.float32)
    output_shape8 = np.array([1, 8, 3]).astype(np.int32)
    strides8 = 1
    padding8 = 'VALID'
    data_format8 = 'NCW'
    dilations8 = 2
    name8 = 'transpose_conv1d_8'

    input_dict8 = {
        'input': tf.convert_to_tensor(input8),
        'filters': tf.convert_to_tensor(filters8),
        'output_shape': tf.convert_to_tensor(output_shape8),
        'strides': strides8,
        'padding': padding8,
        'data_format': data_format8,
        'dilations': dilations8,
        'name': name8
    }
    input_dict8['input'] = tf.transpose(input_dict8['input'], perm=[0, 2, 1])
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.random.rand(3, 12, 2).astype(np.float32)
    filters9 = np.random.rand(4, 4, 2).astype(np.float32)
    output_shape9 = np.array([3, 15, 4]).astype(np.int32)
    strides9 = 1
    padding9 = 'SAME'
    data_format9 = 'NCW'
    dilations9 = 3
    name9 = 'transpose_conv1d_9'

    input_dict9 = {
        'input': tf.convert_to_tensor(input9),
        'filters': tf.convert_to_tensor(filters9),
        'output_shape': tf.convert_to_tensor(output_shape9),
        'strides': strides9,
        'padding': padding9,
        'data_format': data_format9,
        'dilations': dilations9,
        'name': name9
    }
    input_dict9['input'] = tf.transpose(input_dict9['input'], perm=[0, 2, 1])
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.rand(2, 11, 4).astype(np.float32)
    filters10 = np.random.rand(5, 6, 4).astype(np.float32)
    output_shape10 = np.array([2, 15, 6]).astype(np.int32)
    strides10 = 2
    padding10 = 'VALID'
    data_format10 = 'NCW'
    dilations10 = 1
    name10 = 'transpose_conv1d_10'

    input_dict10 = {
        'input': tf.convert_to_tensor(input10),
        'filters': tf.convert_to_tensor(filters10),
        'output_shape': tf.convert_to_tensor(output_shape10),
        'strides': strides10,
        'padding': padding10,
        'data_format': data_format10,
        'dilations': dilations10,
        'name': name10
    }
    input_dict10['input'] = tf.transpose(input_dict10['input'], perm=[0, 2, 1])
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv1d_transpose"] = tf_nn_conv1d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv1d_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_transpose'.")

check_valid('tf.nn.conv1d_transpose', generated_inputs['tf.nn.conv1d_transpose'], lib="tf", suffix=0)
