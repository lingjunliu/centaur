
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_identity_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": "identity_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with float values
    input_tensor = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "identity_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with complex numbers
    input_tensor = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "identity_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with negative values
    input_tensor = np.array([-1, -2, -3], dtype=np.int64)
    input_dict = {"input": input_tensor, "name": "identity_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with boolean values
    input_tensor = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"input": input_tensor, "name": "identity_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tensor
    input_tensor = np.array([], dtype=np.float64)
    input_dict = {"input": input_tensor, "name": "identity_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher dimensional tensor
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {"input": input_tensor, "name": "identity_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with different dtype
    input_tensor = np.array([1, 2, 3], dtype=np.uint8)
    input_dict = {"input": input_tensor, "name": "identity_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar tensor
    input_tensor = np.array(10, dtype=np.int32)
    input_dict = {"input": input_tensor, "name": "identity_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero tensor
    input_tensor = np.zeros((2, 2), dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "identity_10"}
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
