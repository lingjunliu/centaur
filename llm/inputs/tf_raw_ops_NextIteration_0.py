
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_next_iteration_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 tensor
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "float_tensor_1"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor with a different name
    data = np.array([4, 5, 6], dtype=np.int32)
    name = "int_tensor_1"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    data = np.array([[1, 2], [3, 4]], dtype=np.int64)
    name = "2d_int_tensor"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean tensor
    data = np.array([True, False, True], dtype=np.bool_)
    name = "bool_tensor"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String tensor
    data = np.array(["hello", "world"], dtype=np.string_)
    name = "string_tensor"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64 tensor
    data = np.array([1+1j, 2+2j], dtype=np.complex64)
    name = "complex_tensor"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty tensor
    data = np.array([], dtype=np.float32)
    name = "empty_tensor"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with negative values
    data = np.array([-1, -2, -3], dtype=np.int32)
    name = "negative_tensor"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    name = "3d_tensor"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: Uint8 tensor
    data = np.array([10, 20, 30], dtype=np.uint8)
    name = "uint8_tensor"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
temp_list = tf_raw_ops_next_iteration_inputs()
final_list = []

for item in temp_list:
  final_dict = {}
  final_dict["kwargs"] = {}
  final_dict["kwargs"]["data"] = tf.convert_to_tensor(item["data"])
  final_dict["kwargs"]["name"] = item["name"]
  final_list.append(final_dict)

generated_inputs["tf.raw_ops.NextIteration"] = final_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NextIteration' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NextIteration'.")

check_valid('tf.raw_ops.NextIteration', generated_inputs['tf.raw_ops.NextIteration'], lib="tf", suffix=0)
