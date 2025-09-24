
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_identity_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_tensor = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)).numpy()
    name = "float_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor with a name
    input_tensor = tf.constant(np.array([4, 5, 6], dtype=np.int32)).numpy()
    name = "int_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values, different shape
    input_tensor = tf.constant(np.array([[-1, 2], [-3, 4]], dtype=np.int64)).numpy()
    name = "negative_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with zeros
    input_tensor = tf.constant(np.array([0, 0, 0], dtype=np.float64)).numpy()
    name = "zero_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Bool tensor
    input_tensor = tf.constant(np.array([True, False, True], dtype=np.bool_)).numpy()
    name = "bool_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor
    input_tensor = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)).numpy()
    name = "3d_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty tensor
    input_tensor = tf.constant(np.array([], dtype=np.float32)).numpy()
    name = "empty_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 tensor
    input_tensor = tf.constant(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)).numpy()
    name = "complex_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: tf.Variable
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    input_tensor = var.read_value().numpy()
    name = "variable_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String tensor
    input_tensor = tf.constant(np.array(["hello", "world"])).numpy()
    name = "string_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.identity"] = tf_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity'.")

check_valid('tf.identity', generated_inputs['tf.identity'], lib="tf", suffix=0)
