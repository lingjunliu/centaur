
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ensure_shape_inputs():
    list_of_inputs = []

    # Input 1: Valid, simple case
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shape = [2, 3]
    name = "input_tensor"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid, using None for unknown dimension
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shape = [None, 3]
    name = "unknown_dim"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid, 1D tensor
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    shape = [5]
    name = "one_dimensional"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid, 3D tensor
    x = np.random.rand(2, 3, 4).astype(np.float32)
    shape = [2, 3, 4]
    name = "three_dimensional"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid, different data type
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    shape = [2, 3]
    name = "float_tensor"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid, unknown dimensions
    x = np.random.rand(5, 7, 2).astype(np.float32)
    shape = [5, None, 2]
    name = "float_tensor_none"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Valid, using tf.float64
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    shape = [2, 2]
    name = "float64_tensor"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Valid, using tf.string
    x = np.array([["a", "b"], ["c", "d"]]).astype(np.str_)
    shape = [2, 2]
    name = "string_tensor"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Valid, using tf.bool
    x = np.array([[True, False], [False, True]])
    shape = [2, 2]
    name = "bool_tensor"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid, using tf.int64
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    shape = [2, 2]
    name = "int64_tensor"
    input_dict = {"x": tf.convert_to_tensor(x), "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ensure_shape"] = tf_ensure_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ensure_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ensure_shape'.")

check_valid('tf.ensure_shape', generated_inputs['tf.ensure_shape'], lib="tf", suffix=0)
