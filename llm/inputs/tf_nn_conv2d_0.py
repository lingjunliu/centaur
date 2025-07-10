
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv2d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input1 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters1 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides1 = [1, 1, 1, 1]
    padding1 = 'VALID'
    data_format1 = 'NHWC'
    dilations1 = [1, 1, 1, 1]
    name1 = 'conv2d_1'

    input_dict1 = {
        'input': tf.constant(input1).numpy(),
        'filters': tf.constant(filters1).numpy(),
        'strides': strides1,
        'padding': padding1,
        'data_format': data_format1,
        'dilations': dilations1,
        'name': name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: SAME padding
    input2 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters2 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides2 = [1, 1, 1, 1]
    padding2 = 'SAME'
    data_format2 = 'NHWC'
    dilations2 = [1, 1, 1, 1]
    name2 = 'conv2d_2'

    input_dict2 = {
        'input': tf.constant(input2).numpy(),
        'filters': tf.constant(filters2).numpy(),
        'strides': strides2,
        'padding': padding2,
        'data_format': data_format2,
        'dilations': dilations2,
        'name': name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different strides
    input3 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters3 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides3 = [1, 2, 2, 1]
    padding3 = 'VALID'
    data_format3 = 'NHWC'
    dilations3 = [1, 1, 1, 1]
    name3 = 'conv2d_3'

    input_dict3 = {
        'input': tf.constant(input3).numpy(),
        'filters': tf.constant(filters3).numpy(),
        'strides': strides3,
        'padding': padding3,
        'data_format': data_format3,
        'dilations': dilations3,
        'name': name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different data format (NCHW)
    input4 = np.random.rand(1, 3, 5, 5).astype(np.float32)
    filters4 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides4 = [1, 1, 1, 1]
    padding4 = 'VALID'
    data_format4 = 'NCHW'
    dilations4 = [1, 1, 1, 1]
    name4 = 'conv2d_4'

    input_dict4 = {
        'input': tf.constant(input4).numpy(),
        'filters': tf.constant(filters4).numpy(),
        'strides': strides4,
        'padding': padding4,
        'data_format': data_format4,
        'dilations': dilations4,
        'name': name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different dilations
    input5 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters5 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides5 = [1, 1, 1, 1]
    padding5 = 'VALID'
    data_format5 = 'NHWC'
    dilations5 = [1, 2, 2, 1]
    name5 = 'conv2d_5'

    input_dict5 = {
        'input': tf.constant(input5).numpy(),
        'filters': tf.constant(filters5).numpy(),
        'strides': strides5,
        'padding': padding5,
        'data_format': data_format5,
        'dilations': dilations5,
        'name': name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Large Input and Filter
    input6 = np.random.rand(1, 20, 20, 3).astype(np.float32)
    filters6 = np.random.rand(5, 5, 3, 10).astype(np.float32)
    strides6 = [1, 1, 1, 1]
    padding6 = 'VALID'
    data_format6 = 'NHWC'
    dilations6 = [1, 1, 1, 1]
    name6 = 'conv2d_6'

    input_dict6 = {
        'input': tf.constant(input6).numpy(),
        'filters': tf.constant(filters6).numpy(),
        'strides': strides6,
        'padding': padding6,
        'data_format': data_format6,
        'dilations': dilations6,
        'name': name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Batch size > 1
    input7 = np.random.rand(4, 5, 5, 3).astype(np.float32)
    filters7 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides7 = [1, 1, 1, 1]
    padding7 = 'VALID'
    data_format7 = 'NHWC'
    dilations7 = [1, 1, 1, 1]
    name7 = 'conv2d_7'

    input_dict7 = {
        'input': tf.constant(input7).numpy(),
        'filters': tf.constant(filters7).numpy(),
        'strides': strides7,
        'padding': padding7,
        'data_format': data_format7,
        'dilations': dilations7,
        'name': name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

   # Input 8: filter size 1x1
    input8 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters8 = np.random.rand(1, 1, 3, 2).astype(np.float32)
    strides8 = [1, 1, 1, 1]
    padding8 = 'VALID'
    data_format8 = 'NHWC'
    dilations8 = [1, 1, 1, 1]
    name8 = 'conv2d_8'

    input_dict8 = {
        'input': tf.constant(input8).numpy(),
        'filters': tf.constant(filters8).numpy(),
        'strides': strides8,
        'padding': padding8,
        'data_format': data_format8,
        'dilations': dilations8,
        'name': name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: in_channels = out_channels = 1
    input9 = np.random.rand(1, 5, 5, 1).astype(np.float32)
    filters9 = np.random.rand(3, 3, 1, 1).astype(np.float32)
    strides9 = [1, 1, 1, 1]
    padding9 = 'VALID'
    data_format9 = 'NHWC'
    dilations9 = [1, 1, 1, 1]
    name9 = 'conv2d_9'

    input_dict9 = {
        'input': tf.constant(input9).numpy(),
        'filters': tf.constant(filters9).numpy(),
        'strides': strides9,
        'padding': padding9,
        'data_format': data_format9,
        'dilations': dilations9,
        'name': name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: negative values in input
    input10 = np.random.rand(1, 5, 5, 3).astype(np.float32) - 0.5
    filters10 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides10 = [1, 1, 1, 1]
    padding10 = 'VALID'
    data_format10 = 'NHWC'
    dilations10 = [1, 1, 1, 1]
    name10 = 'conv2d_10'

    input_dict10 = {
        'input': tf.constant(input10).numpy(),
        'filters': tf.constant(filters10).numpy(),
        'strides': strides10,
        'padding': padding10,
        'data_format': data_format10,
        'dilations': dilations10,
        'name': name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv2d"] = tf_nn_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv2d'.")

check_valid('tf.nn.conv2d', generated_inputs['tf.nn.conv2d'], lib="tf", suffix=0)
