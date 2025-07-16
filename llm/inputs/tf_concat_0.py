
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_concat_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D concatenation along axis 0
    values = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]
    axis = 0
    name = "concat_example_1"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D concatenation along axis 1
    values = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]
    axis = 1
    name = "concat_example_2"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D concatenation along axis 0
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = 0
    name = "concat_example_3"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D concatenation along axis 1
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = 1
    name = "concat_example_4"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D concatenation along axis 2
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = 2
    name = "concat_example_5"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Concatenation with a single tensor
    values = [np.array([[1, 2], [3, 4]])]
    axis = 1
    name = "concat_example_6"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative axis example
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = -1
    name = "concat_example_7"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Negative axis example with 2D array
    values = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = -1
    name = "concat_example_8"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  4D tensor concatenation
    values = [np.random.rand(2, 3, 4, 5), np.random.rand(2, 3, 4, 5)]
    axis = 2
    name = "concat_example_9"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Concatenating more than 2 tensors
    values = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
    axis = 1
    name = "concat_example_10"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_inputs = tf_concat_inputs()
for i in range(len(temp_inputs)):
    temp_input = temp_inputs[i]
    values_list = temp_input["values"]
    tensor_list = [tf.convert_to_tensor(arr, dtype=tf.float32) for arr in values_list]
    temp_input["values"] = tensor_list
    temp_inputs[i] = temp_input

generated_inputs["tf.concat"] = temp_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.concat'.")

check_valid('tf.concat', generated_inputs['tf.concat'], lib="tf", suffix=0)
