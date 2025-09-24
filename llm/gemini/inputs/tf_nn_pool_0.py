
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_pool_inputs():
    list_of_inputs = []

    # Input 1: 2D input, MAX pooling, VALID padding
    input1 = np.random.rand(1, 8, 8, 3).astype(np.float32)
    window_shape1 = [2, 2]
    pooling_type1 = "MAX"
    strides1 = [2, 2]
    padding1 = "VALID"
    data_format1 = "NHWC"
    dilations1 = [1, 1]
    name1 = "max_pool_1"

    input_dict1 = {
        "input": input1,
        "window_shape": window_shape1,
        "pooling_type": pooling_type1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "dilations": dilations1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D input, AVG pooling, SAME padding
    input2 = np.random.rand(1, 8, 8, 3).astype(np.float32)
    window_shape2 = [3, 3]
    pooling_type2 = "AVG"
    strides2 = [1, 1]
    padding2 = "SAME"
    data_format2 = "NHWC"
    dilations2 = [1, 1]
    name2 = "avg_pool_1"

    input_dict2 = {
        "input": input2,
        "window_shape": window_shape2,
        "pooling_type": pooling_type2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "dilations": dilations2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D input, MAX pooling, VALID padding
    input3 = np.random.rand(1, 4, 4, 4, 3).astype(np.float32)
    window_shape3 = [2, 2, 2]
    pooling_type3 = "MAX"
    strides3 = [2, 2, 2]
    padding3 = "VALID"
    data_format3 = "NDHWC"
    dilations3 = [1, 1, 1]
    name3 = "max_pool_3d"

    input_dict3 = {
        "input": input3,
        "window_shape": window_shape3,
        "pooling_type": pooling_type3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "dilations": dilations3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D input, AVG pooling, SAME padding
    input4 = np.random.rand(1, 4, 4, 4, 3).astype(np.float32)
    window_shape4 = [3, 3, 3]
    pooling_type4 = "AVG"
    strides4 = [1, 1, 1]
    padding4 = "SAME"
    data_format4 = "NDHWC"
    dilations4 = [1, 1, 1]
    name4 = "avg_pool_3d"

    input_dict4 = {
        "input": input4,
        "window_shape": window_shape4,
        "pooling_type": pooling_type4,
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "dilations": dilations4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D input, MAX pooling, SAME padding, NCHW
    input5 = np.random.rand(1, 3, 8, 8).astype(np.float32)
    window_shape5 = [2, 2]
    pooling_type5 = "MAX"
    strides5 = [2, 2]
    padding5 = "SAME"
    data_format5 = "NCHW"
    dilations5 = [1, 1]
    name5 = "max_pool_nchw"

    input_dict5 = {
        "input": input5,
        "window_shape": window_shape5,
        "pooling_type": pooling_type5,
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "dilations": dilations5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D input, AVG pooling, VALID padding, NCHW
    input6 = np.random.rand(1, 3, 8, 8).astype(np.float32)
    window_shape6 = [3, 3]
    pooling_type6 = "AVG"
    strides6 = [1, 1]
    padding6 = "VALID"
    data_format6 = "NCHW"
    dilations6 = [1, 1]
    name6 = "avg_pool_nchw"

    input_dict6 = {
        "input": input6,
        "window_shape": window_shape6,
        "pooling_type": pooling_type6,
        "strides": strides6,
        "padding": padding6,
        "data_format": data_format6,
        "dilations": dilations6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

   # Input 7: 1D input, MAX pooling, VALID padding
    input7 = np.random.rand(1, 16, 3).astype(np.float32)
    window_shape7 = [2]
    pooling_type7 = "MAX"
    strides7 = [2]
    padding7 = "VALID"
    data_format7 = "NWC"
    dilations7 = [1]
    name7 = "max_pool_1d"

    input_dict7 = {
        "input": input7,
        "window_shape": window_shape7,
        "pooling_type": pooling_type7,
        "strides": strides7,
        "padding": padding7,
        "data_format": data_format7,
        "dilations": dilations7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 1D input, AVG pooling, SAME padding
    input8 = np.random.rand(1, 16, 3).astype(np.float32)
    window_shape8 = [3]
    pooling_type8 = "AVG"
    strides8 = [1]
    padding8 = "SAME"
    data_format8 = "NWC"
    dilations8 = [1]
    name8 = "avg_pool_1d"

    input_dict8 = {
        "input": input8,
        "window_shape": window_shape8,
        "pooling_type": pooling_type8,
        "strides": strides8,
        "padding": padding8,
        "data_format": data_format8,
        "dilations": dilations8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 1D input, MAX pooling, SAME padding, NCW
    input9 = np.random.rand(1, 3, 16).astype(np.float32)
    window_shape9 = [2]
    pooling_type9 = "MAX"
    strides9 = [2]
    padding9 = "SAME"
    data_format9 = "NCW"
    dilations9 = [1]
    name9 = "max_pool_ncw"

    input_dict9 = {
        "input": input9,
        "window_shape": window_shape9,
        "pooling_type": pooling_type9,
        "strides": strides9,
        "padding": padding9,
        "data_format": data_format9,
        "dilations": dilations9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 2D input, AVG pooling, VALID padding, dilations
    input10 = np.random.rand(1, 8, 8, 3).astype(np.float32)
    window_shape10 = [2, 2]
    pooling_type10 = "AVG"
    strides10 = [1, 1]
    padding10 = "VALID"
    data_format10 = "NHWC"
    dilations10 = [2, 2]
    name10 = "avg_pool_dilated"

    input_dict10 = {
        "input": input10,
        "window_shape": window_shape10,
        "pooling_type": pooling_type10,
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "dilations": dilations10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.pool"] = tf_nn_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.pool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.pool'.")

check_valid('tf.nn.pool', generated_inputs['tf.nn.pool'], lib="tf", suffix=0)
