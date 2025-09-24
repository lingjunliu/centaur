
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_identity_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensor
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.int32).numpy(), "name": "identity_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor with a different shape
    input_tensor = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.float32).numpy(), "name": "identity_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String tensor
    input_tensor = np.array(["hello", "world"], dtype=np.string_)
    input_dict = {"input": tf.constant(input_tensor).numpy(), "name": "identity_string"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean tensor
    input_tensor = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.bool).numpy(), "name": "identity_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative values
    input_tensor = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.int32).numpy(), "name": "identity_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher dimensional tensor (3D)
    input_tensor = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.float64).numpy(), "name": "identity_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty tensor
    input_tensor = np.array([], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.int32).numpy(), "name": "identity_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Complex64 Tensor
    input_tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.complex64).numpy(), "name": "identity_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 0 Tensor (scalar)
    input_tensor = np.array(10, dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.int32).numpy(), "name": "identity_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Identity"] = tf_raw_ops_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Identity'.")

check_valid('tf.raw_ops.Identity', generated_inputs['tf.raw_ops.Identity'], lib="tf", suffix=0)
