
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_guarantee_const_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor of integers
    input_tensor = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    name = "input_tensor_1"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor of floats
    input_tensor = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    name = "input_tensor_2"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor of complex numbers
    input_tensor = tf.constant(np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64))
    name = "input_tensor_3"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with negative values
    input_tensor = tf.constant(np.array([-1, -2, -3, -4, -5], dtype=np.int32))
    name = "input_tensor_4"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with zeros
    input_tensor = tf.constant(np.array([0, 0, 0, 0, 0], dtype=np.int32))
    name = "input_tensor_5"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with mixed positive and negative values
    input_tensor = tf.constant(np.array([-1, 2, -3, 4, -5], dtype=np.int32))
    name = "input_tensor_6"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor
    input_tensor = tf.constant(np.random.rand(2, 3, 4, 5).astype(np.float32))
    name = "input_tensor_7"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with boolean values
    input_tensor = tf.constant(np.array([True, False, True, False], dtype=np.bool_))
    name = "input_tensor_8"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with different data type (int64)
    input_tensor = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    name = "input_tensor_9"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty tensor
    input_tensor = tf.constant(np.array([], dtype=np.float32))
    name = "input_tensor_10"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
inputs = tf_guarantee_const_inputs()
for i in range(len(inputs)):
  inputs[i]["input"] = inputs[i]["input"].numpy()

generated_inputs["tf.guarantee_const"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.guarantee_const' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.guarantee_const'.")

check_valid('tf.guarantee_const', generated_inputs['tf.guarantee_const'], lib="tf", suffix=0)
