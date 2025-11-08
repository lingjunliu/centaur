
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_convert_to_tensor_inputs():
    list_of_inputs = []
    
    # Input 1: Tensor with int32 dtype
    value = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    dtype = tf.int32
    dtype_hint = tf.int32
    name = "test_tensor"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Tensor with float32 dtype
    value = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test_float"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Tensor with bool dtype
    value = tf.constant([True, False, True])
    dtype = tf.bool
    dtype_hint = tf.bool
    name = "test_bool"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Numpy array with int32 dtype
    value = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dtype = tf.int32
    dtype_hint = tf.int32
    name = "test_numpy_int"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Numpy array with float64 dtype
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dtype = tf.float64
    dtype_hint = tf.float64
    name = "test_numpy_float"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Numpy array with bool dtype (using bool_)
    value = np.array([True, False, True], dtype=np.bool_)
    dtype = tf.bool
    dtype_hint = tf.bool
    name = "test_numpy_bool"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Numpy array with complex64 dtype
    value = np.array([1+2j, 3+4j], dtype=np.complex64)
    dtype = tf.complex64
    dtype_hint = tf.complex64
    name = "test_numpy_complex"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Numpy array with string dtype
    value = np.array(["hello", "world"], dtype=np.str_)
    dtype = tf.string
    dtype_hint = tf.string
    name = "test_numpy_string"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Numpy array with float32 dtype
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test_numpy_float_2"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Tensor with string dtype
    value = tf.constant(["hello", "world"])
    dtype = tf.string
    dtype_hint = tf.string
    name = "test_string_2"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.convert_to_tensor"] = tf_convert_to_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.convert_to_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.convert_to_tensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.convert_to_tensor', generated_inputs['tf.convert_to_tensor'], lib="tf", suffix=0)
