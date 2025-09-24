
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input1 = np.random.rand(1, 10, 1).astype(np.float32)
    ksize1 = [1]
    strides1 = [1]
    padding1 = "VALID"
    data_format1 = "NWC"
    name1 = "maxpool1"
    input_dict1 = {"input": input1, "ksize": ksize1, "strides": strides1, "padding": padding1, "data_format": data_format1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different ksize and stride
    input2 = np.random.rand(1, 15, 1).astype(np.float32)
    ksize2 = [3]
    strides2 = [2]
    padding2 = "VALID"
    data_format2 = "NWC"
    name2 = "maxpool2"
    input_dict2 = {"input": input2, "ksize": ksize2, "strides": strides2, "padding": padding2, "data_format": data_format2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: SAME padding
    input3 = np.random.rand(1, 10, 1).astype(np.float32)
    ksize3 = [3]
    strides3 = [1]
    padding3 = "SAME"
    data_format3 = "NWC"
    name3 = "maxpool3"
    input_dict3 = {"input": input3, "ksize": ksize3, "strides": strides3, "padding": padding3, "data_format": data_format3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multiple channels
    input4 = np.random.rand(1, 10, 3).astype(np.float32)
    ksize4 = [2]
    strides4 = [2]
    padding4 = "VALID"
    data_format4 = "NWC"
    name4 = "maxpool4"
    input_dict4 = {"input": input4, "ksize": ksize4, "strides": strides4, "padding": padding4, "data_format": data_format4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multiple examples
    input5 = np.random.rand(4, 10, 1).astype(np.float32)
    ksize5 = [2]
    strides5 = [2]
    padding5 = "VALID"
    data_format5 = "NWC"
    name5 = "maxpool5"
    input_dict5 = {"input": input5, "ksize": ksize5, "strides": strides5, "padding": padding5, "data_format": data_format5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: NCW data format
    input6 = np.random.rand(1, 1, 10).astype(np.float32)
    ksize6 = [2]
    strides6 = [2]
    padding6 = "VALID"
    data_format6 = "NCW"
    name6 = "maxpool6"
    input_dict6 = {"input": input6, "ksize": ksize6, "strides": strides6, "padding": padding6, "data_format": data_format6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Negative values
    input7 = np.random.randn(1, 10, 1).astype(np.float32)
    ksize7 = [2]
    strides7 = [2]
    padding7 = "VALID"
    data_format7 = "NWC"
    name7 = "maxpool7"
    input_dict7 = {"input": input7, "ksize": ksize7, "strides": strides7, "padding": padding7, "data_format": data_format7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Explicit Padding
    input8 = np.random.rand(1, 5, 1).astype(np.float32)
    ksize8 = [2]
    strides8 = [1]
    padding8 = [[0, 0], [1, 1], [0, 0]]
    data_format8 = "NWC"
    name8 = "maxpool8"
    input_dict8 = {"input": input8, "ksize": ksize8, "strides": strides8, "padding": padding8, "data_format": data_format8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

     # Input 9: Ksize and strides as int
    input9 = np.random.rand(1, 10, 1).astype(np.float32)
    ksize9 = [2]
    strides9 = [2]
    padding9 = "VALID"
    data_format9 = "NWC"
    name9 = "maxpool9"
    input_dict9 = {"input": input9, "ksize": ksize9, "strides": strides9, "padding": padding9, "data_format": data_format9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Larger Input Size
    input10 = np.random.rand(2, 25, 4).astype(np.float32)
    ksize10 = [4]
    strides10 = [3]
    padding10 = "SAME"
    data_format10 = "NWC"
    name10 = "maxpool10"
    input_dict10 = {"input": input10, "ksize": ksize10, "strides": strides10, "padding": padding10, "data_format": data_format10, "name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.max_pool1d"] = tf_nn_max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.max_pool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool1d'.")

check_valid('tf.nn.max_pool1d', generated_inputs['tf.nn.max_pool1d'], lib="tf", suffix=0)
