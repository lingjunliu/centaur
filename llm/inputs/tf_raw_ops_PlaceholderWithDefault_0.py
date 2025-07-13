
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_PlaceholderWithDefault_inputs():
    list_of_inputs = []

    # Input 1, valid
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    shape = input_tensor.shape.tolist()
    name = "placeholder_1"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    shape = input_tensor.shape.tolist()
    name = "placeholder_2"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input_tensor = np.array([True, False, True], dtype=np.bool_)
    shape = input_tensor.shape.tolist()
    name = "placeholder_3"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input_tensor = np.array(["hello", "world"], dtype=np.string_)
    shape = input_tensor.shape.tolist()
    name = "placeholder_4"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, 3D tensor
    input_tensor = np.random.rand(2, 3, 4).astype(np.float64)
    shape = input_tensor.shape.tolist()
    name = "placeholder_5"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, scalar tensor
    input_tensor = np.array(5, dtype=np.int64)
    shape = []
    name = "placeholder_6"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, negative values
    input_tensor = np.array([-1, -2, -3], dtype=np.int32)
    shape = input_tensor.shape.tolist()
    name = "placeholder_7"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, complex values
    input_tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    shape = input_tensor.shape.tolist()
    name = "placeholder_8"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, empty array
    input_tensor = np.array([], dtype=np.int32)
    shape = [0]
    name = "placeholder_9"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10, valid, large shape
    input_tensor = np.random.rand(100, 100).astype(np.float32)
    shape = input_tensor.shape.tolist()
    name = "placeholder_10"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.PlaceholderWithDefault"] = tf_raw_ops_PlaceholderWithDefault_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.PlaceholderWithDefault' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PlaceholderWithDefault'.")

check_valid('tf.raw_ops.PlaceholderWithDefault', generated_inputs['tf.raw_ops.PlaceholderWithDefault'], lib="tf", suffix=0)
