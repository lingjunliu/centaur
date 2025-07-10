
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_full_like_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    fill_value = tf.constant(7)
    dtype = np.int32
    order = 'K'
    subok = True
    shape = (2,3)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float values
    a = tf.constant([[1.1, 2.2], [3.3, 4.4]])
    fill_value = tf.constant(5.5)
    dtype = np.float64
    order = 'K'
    subok = True
    shape = (2,2)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boolean values
    a = tf.constant([[True, False], [False, True]])
    fill_value = tf.constant(True)
    dtype = np.bool_
    order = 'K'
    subok = True
    shape = (2,2)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.int64)
    fill_value = tf.constant(9)
    dtype = np.int32
    order = 'K'
    subok = True
    shape = (2,2)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    fill_value = tf.constant(10)
    dtype = np.int32
    order = 'K'
    subok = True
    shape = (2,2,2)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative fill value
    a = tf.constant([[1, 2], [3, 4]])
    fill_value = tf.constant(-5)
    dtype = np.int32
    order = 'K'
    subok = True
    shape = (2,2)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: different shape
    a = tf.constant([[1, 2], [3, 4]])
    fill_value = tf.constant(1)
    dtype = np.int32
    order = 'K'
    subok = True
    shape = (3, 2)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    fill_value = tf.constant(2.5, dtype=tf.float32)
    dtype = np.float16
    order = 'K'
    subok = True
    shape = (2,2)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex
    a = tf.constant([[1+1j, 2+2j], [3+3j, 4+4j]])
    fill_value = tf.constant(5+5j)
    dtype = np.complex64
    order = 'K'
    subok = True
    shape = (2,2)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: all zeros
    a = tf.zeros((2, 2), dtype=tf.int32)
    fill_value = tf.constant(1)
    dtype = np.int32
    order = 'K'
    subok = True
    shape = (2,2)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.full_like"] = tf_experimental_numpy_full_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.full_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.full_like'.")

check_valid('tf.experimental.numpy.full_like', generated_inputs['tf.experimental.numpy.full_like'], lib="tf", suffix=0)
