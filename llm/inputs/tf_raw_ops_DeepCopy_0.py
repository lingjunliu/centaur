
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_deepcopy_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensor
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {"x": x, "name": "int_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor
    x = np.array([1.0, 2.5, 3.7, 4.2, 5.9], dtype=np.float32)
    input_dict = {"x": x, "name": "float_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boolean tensor
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {"x": x, "name": "bool_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String tensor
    x = np.array(["hello", "world", "tensorflow"], dtype=np.string_)
    input_dict = {"x": x, "name": "string_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D integer tensor
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    input_dict = {"x": x, "name": "2d_int_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float tensor
    x = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict = {"x": x, "name": "3d_float_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with negative values
    x = np.array([-1, -2, 0, 1, 2], dtype=np.int16)
    input_dict = {"x": x, "name": "negative_int_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tensor
    x = np.array([], dtype=np.float16)
    input_dict = {"x": x, "name": "empty_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex tensor
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"x": x, "name": "complex_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Large integer tensor
    x = np.array([2**31-1, -(2**31)], dtype=np.int64)
    input_dict = {"x": x, "name": "large_int_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_deepcopy_inputs()
generated_inputs["tf.raw_ops.DeepCopy"] = []
for input_dict in inputs:
    x = tf.convert_to_tensor(input_dict['x'])
    name = input_dict['name']
    generated_inputs["tf.raw_ops.DeepCopy"].append({"x": x, "name": name})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DeepCopy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeepCopy'.")

check_valid('tf.raw_ops.DeepCopy', generated_inputs['tf.raw_ops.DeepCopy'], lib="tf", suffix=0)
