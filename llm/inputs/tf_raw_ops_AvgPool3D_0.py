
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_avgpool3d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input1 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    ksize1 = [1, 2, 2, 2, 1]
    strides1 = [1, 1, 1, 1, 1]
    padding1 = "VALID"
    data_format1 = "NDHWC"

    input_dict = {
        "input": input1,
        "ksize": ksize1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different ksize and strides
    input2 = np.random.rand(1, 10, 10, 10, 3).astype(np.float32)
    ksize2 = [1, 3, 3, 3, 1]
    strides2 = [1, 2, 2, 2, 1]
    padding2 = "SAME"
    data_format2 = "NDHWC"

    input_dict = {
        "input": input2,
        "ksize": ksize2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NCDHW data format
    input3 = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    ksize3 = [1, 1, 3, 3, 3]
    strides3 = [1, 1, 2, 2, 2]
    padding3 = "VALID"
    data_format3 = "NCDHW"

    input_dict = {
        "input": input3,
        "ksize": ksize3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32
    input4 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    ksize4 = [1, 2, 2, 2, 1]
    strides4 = [1, 1, 1, 1, 1]
    padding4 = "VALID"
    data_format4 = "NDHWC"

    input_dict = {
        "input": input4,
        "ksize": ksize4,
        "strides": strides4,
        "padding": padding4,
        "data_format": data_format4,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5:  float32 with SAME
    input5 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    ksize5 = [1, 2, 2, 2, 1]
    strides5 = [1, 1, 1, 1, 1]
    padding5 = "SAME"
    data_format5 = "NDHWC"

    input_dict = {
        "input": input5,
        "ksize": ksize5,
        "strides": strides5,
        "padding": padding5,
        "data_format": data_format5,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large ksize and strides
    input6 = np.random.rand(1, 20, 20, 20, 3).astype(np.float32)
    ksize6 = [1, 10, 10, 10, 1]
    strides6 = [1, 5, 5, 5, 1]
    padding6 = "VALID"
    data_format6 = "NDHWC"

    input_dict = {
        "input": input6,
        "ksize": ksize6,
        "strides": strides6,
        "padding": padding6,
        "data_format": data_format6,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Small input size, large ksize
    input7 = np.random.rand(1, 3, 3, 3, 3).astype(np.float32)
    ksize7 = [1, 4, 4, 4, 1]
    strides7 = [1, 1, 1, 1, 1]
    padding7 = "VALID"
    data_format7 = "NDHWC"

    input_dict = {
        "input": input7,
        "ksize": ksize7,
        "strides": strides7,
        "padding": padding7,
        "data_format": data_format7,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: SAME padding with small input and large ksize
    input8 = np.random.rand(1, 3, 3, 3, 3).astype(np.float32)
    ksize8 = [1, 4, 4, 4, 1]
    strides8 = [1, 1, 1, 1, 1]
    padding8 = "SAME"
    data_format8 = "NDHWC"

    input_dict = {
        "input": input8,
        "ksize": ksize8,
        "strides": strides8,
        "padding": padding8,
        "data_format": data_format8,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 input and custom name
    input9 = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    ksize9 = [1, 2, 2, 2, 1]
    strides9 = [1, 1, 1, 1, 1]
    padding9 = "VALID"
    data_format9 = "NDHWC"
    name9 = "my_avg_pool"

    input_dict = {
        "input": input9,
        "ksize": ksize9,
        "strides": strides9,
        "padding": padding9,
        "data_format": data_format9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: NCDHW with SAME
    input10 = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    ksize10 = [1, 1, 3, 3, 3]
    strides10 = [1, 1, 2, 2, 2]
    padding10 = "SAME"
    data_format10 = "NCDHW"

    input_dict = {
        "input": input10,
        "ksize": ksize10,
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AvgPool3D"] = tf_raw_ops_avgpool3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AvgPool3D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AvgPool3D'.")

check_valid('tf.raw_ops.AvgPool3D', generated_inputs['tf.raw_ops.AvgPool3D'], lib="tf", suffix=0)
