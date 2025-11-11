
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_zeros_like_inputs():
    list_of_inputs = []
    
    # Input 1: 2D array with float dtype
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": a, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D array with int dtype
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {"a": a, "dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D array with bool dtype
    a = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"a": a, "dtype": np.bool_}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D array with complex dtype
    a = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {"a": a, "dtype": np.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D array with float64 dtype
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"a": a, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D array with uint8 dtype
    a = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    input_dict = {"a": a, "dtype": np.uint8}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D array with int64 dtype
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"a": a, "dtype": np.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D array with complex128 dtype
    a = np.array([1+2j, 3+4j], dtype=np.complex128)
    input_dict = {"a": a, "dtype": np.complex128}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D array with float32 dtype
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": a, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D array with int32 dtype
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"a": a, "dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.zeros_like"] = tf_zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.zeros_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.zeros_like'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.zeros_like', generated_inputs['tf.experimental.numpy.zeros_like'], lib="tf", suffix=0)
