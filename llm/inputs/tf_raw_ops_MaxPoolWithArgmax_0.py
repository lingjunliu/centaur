
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolWithArgmax_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    ksize1 = [1, 1, 1, 1]
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"
    input_dict = {
        "input": input1,
        "ksize": ksize1,
        "strides": strides1,
        "padding": padding1,
        "Targmax": tf.int64,
        "include_batch_in_index": False,
        "name": "maxpool1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input2 = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    ksize2 = [1, 2, 2, 1]
    strides2 = [1, 2, 2, 1]
    padding2 = "VALID"

    input_dict = {
        "input": input2,
        "ksize": ksize2,
        "strides": strides2,
        "padding": padding2,
        "Targmax": tf.int32,
        "include_batch_in_index": True,
        "name": "maxpool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input3 = np.array([[[[1, 2, 3], [4, 5, 6], [7,8,9]]]], dtype=np.int32)
    ksize3 = [1, 2, 2, 1]
    strides3 = [1, 1, 1, 1]
    padding3 = "SAME"
    input_dict = {
        "input": input3,
        "ksize": ksize3,
        "strides": strides3,
        "padding": padding3,
        "Targmax": tf.int64,
        "include_batch_in_index": False,
        "name": "maxpool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input4 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    ksize4 = [1, 2, 2, 1]
    strides4 = [1, 1, 1, 1]
    padding4 = "SAME"
    input_dict = {
        "input": input4,
        "ksize": ksize4,
        "strides": strides4,
        "padding": padding4,
        "Targmax": tf.int32,
        "include_batch_in_index": True,
        "name": "maxpool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input5 = np.array([[[[-1.0, -2.0], [-3.0, -4.0]]]], dtype=np.float32)
    ksize5 = [1, 1, 1, 1]
    strides5 = [1, 1, 1, 1]
    padding5 = "VALID"
    input_dict = {
        "input": input5,
        "ksize": ksize5,
        "strides": strides5,
        "padding": padding5,
        "Targmax": tf.int64,
        "include_batch_in_index": False,
        "name": "maxpool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input6 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    ksize6 = [1, 1, 1, 1]
    strides6 = [1, 1, 1, 1]
    padding6 = "SAME"
    input_dict = {
        "input": input6,
        "ksize": ksize6,
        "strides": strides6,
        "padding": padding6,
        "Targmax": tf.int32,
        "include_batch_in_index": True,
        "name": "maxpool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input7 = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=np.float64)
    ksize7 = [1, 1, 1, 1]
    strides7 = [1, 1, 1, 1]
    padding7 = "VALID"
    input_dict = {
        "input": input7,
        "ksize": ksize7,
        "strides": strides7,
        "padding": padding7,
        "Targmax": tf.int64,
        "include_batch_in_index": False,
        "name": "maxpool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input8 = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    ksize8 = [1, 1, 1, 1]
    strides8 = [1, 1, 1, 1]
    padding8 = "VALID"
    input_dict = {
        "input": input8,
        "ksize": ksize8,
        "strides": strides8,
        "padding": padding8,
        "Targmax": tf.int32,
        "include_batch_in_index": True,
        "name": "maxpool8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input9 = np.array([[[[1, 2, 3], [4, 5, 6], [7,8,9]]]], dtype=np.int32)
    ksize9 = [1, 1, 1, 1]
    strides9 = [1, 1, 1, 1]
    padding9 = "SAME"
    input_dict = {
        "input": input9,
        "ksize": ksize9,
        "strides": strides9,
        "padding": padding9,
        "Targmax": tf.int64,
        "include_batch_in_index": False,
        "name": "maxpool9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input10 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    ksize10 = [1, 1, 1, 1]
    strides10 = [1, 1, 1, 1]
    padding10 = "SAME"
    input_dict = {
        "input": input10,
        "ksize": ksize10,
        "strides": strides10,
        "padding": padding10,
        "Targmax": tf.int32,
        "include_batch_in_index": True,
        "name": "maxpool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPoolWithArgmax"] = tf_raw_ops_MaxPoolWithArgmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPoolWithArgmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolWithArgmax'.")

check_valid('tf.raw_ops.MaxPoolWithArgmax', generated_inputs['tf.raw_ops.MaxPoolWithArgmax'], lib="tf", suffix=0)
