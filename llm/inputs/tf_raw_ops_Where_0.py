
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_where_inputs():
    list_of_inputs = []

    # Input 1: Simple boolean tensor
    condition = np.array([[True, False], [False, True]])
    name = None
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D boolean tensor
    condition = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    name = "where_op_2"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float32 tensor (non-zero elements are treated as True)
    condition = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    name = "where_op_3"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int32 tensor (non-zero elements are treated as True)
    condition = np.array([[1, 0], [0, -1]], dtype=np.int32)
    name = "where_op_4"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty boolean tensor
    condition = np.array([], dtype=np.bool_)
    condition = condition.reshape((0,2))
    name = "where_op_5"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D boolean tensor
    condition = np.array([True, False, True], dtype=np.bool_)
    name = "where_op_6"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_inputs = tf_raw_ops_where_inputs()
converted_inputs = []
for inp in temp_inputs:
    converted_inp = {}
    condition = inp["condition"]
    if condition.size > 0:
        condition = tf.convert_to_tensor(condition)
    else:
        condition = tf.convert_to_tensor(condition, dtype=np.bool_)

    converted_inp["condition"] = condition
    converted_inp["name"] = inp["name"]
    converted_inputs.append(converted_inp)
generated_inputs["tf.raw_ops.Where"] = converted_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Where' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Where'.")

check_valid('tf.raw_ops.Where', generated_inputs['tf.raw_ops.Where'], lib="tf", suffix=0)
