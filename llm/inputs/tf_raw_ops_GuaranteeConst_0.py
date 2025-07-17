
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_GuaranteeConst_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": "const_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with float values
    input_tensor = np.array([[1.0, 2.5], [3.2, 4.7]], dtype=np.float32)
    input_dict = {"input": tf.constant(input_tensor), "name": "const_2d_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with complex values
    input_tensor = np.array([[[1+1j, 2-2j], [3+0j, 4-1j]]], dtype=np.complex64)
    input_dict = {"input": tf.constant(input_tensor), "name": "const_3d_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with boolean values
    input_tensor = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"input": tf.constant(input_tensor), "name": "const_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with string values
    input_tensor = np.array(["hello", "world"], dtype=np.string_)
    input_dict = {"input": tf.constant(input_tensor), "name": "const_string"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tensor
    input_tensor = np.array([], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.int32, shape=(0,)), "name": "const_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with negative values
    input_tensor = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": "const_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with large values
    input_tensor = np.array([2**15 - 1, 2**15 - 2], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": "const_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with zeros
    input_tensor = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": "const_zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher rank tensor
    input_tensor = np.random.rand(2,3,4).astype(np.float32)
    input_dict = {"input": tf.constant(input_tensor), "name": "const_high_rank"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GuaranteeConst"] = tf_raw_ops_GuaranteeConst_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GuaranteeConst' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GuaranteeConst'.")

check_valid('tf.raw_ops.GuaranteeConst', generated_inputs['tf.raw_ops.GuaranteeConst'], lib="tf", suffix=0)
