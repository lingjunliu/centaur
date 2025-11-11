
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []
    
    # Input 1: tensor with dtype specified
    val = tf.constant([1, 2, 3])
    dtype = tf.int32
    copy = True
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: tensor with different dtype, no copy needed
    val = tf.constant([1.0, 2.0, 3.0])
    dtype = tf.float32
    copy = False
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: tensor with ndmin = 1
    val = tf.constant([1, 2, 3])
    dtype = None
    copy = True
    ndmin = 1
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: tensor with ndmin = 2
    val = tf.constant([1, 2, 3])
    dtype = None
    copy = False
    ndmin = 2
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: tensor with negative values
    val = tf.constant([-1, -2, -3])
    dtype = None
    copy = True
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: tensor with float values and copy=True
    val = tf.constant([1.5, 2.7, 3.9])
    dtype = tf.float64
    copy = True
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: tensor with complex values
    val = tf.constant([1+2j, 3+4j])
    dtype = tf.complex64
    copy = False
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: tensor with ndmin = 3
    val = tf.constant([[[1, 2], [3, 4]]])
    dtype = None
    copy = True
    ndmin = 3
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: tensor with zero values
    val = tf.constant([0, 0, 0])
    dtype = None
    copy = False
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: tensor with mixed dimensions
    val = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dtype = None
    copy = True
    ndmin = 1
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array"] = tf_experimental_numpy_array_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.array' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.array'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.array', generated_inputs['tf.experimental.numpy.array'], lib="tf", suffix=0)
