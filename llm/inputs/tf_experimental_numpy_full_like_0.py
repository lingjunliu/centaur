
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_full_like_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[1, 2], [3, 4]])
    fill_value = np.array(5)
    dtype = np.int32
    order = 'K'
    subok = True
    shape = None

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([1, 2, 3])
    fill_value = np.array(2.5)
    dtype = np.float64
    order = 'K'
    subok = True
    shape = None

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    fill_value = np.array(-1)
    dtype = np.int64
    order = 'K'
    subok = True
    shape = None

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([1])
    fill_value = np.array(0)
    dtype = np.bool_
    order = 'K'
    subok = True
    shape = None

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([])
    fill_value = np.array(100)
    dtype = np.int32
    order = 'K'
    subok = True
    shape = None

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([1, 2, 3], dtype=np.int16)
    fill_value = np.array(5, dtype=np.int16)
    dtype = np.int16
    order = 'K'
    subok = True
    shape = None

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[1.1, 2.2], [3.3, 4.4]])
    fill_value = np.array(6.6)
    dtype = np.float32
    order = 'K'
    subok = True
    shape = None

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    a = np.array([1, 2, 3])
    fill_value = np.array(0)
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

    # Input 9
    a = np.array([[1, 2], [3, 4]])
    fill_value = np.array(-100)
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

    # Input 10
    a = np.array([1.0, 2.0, 3.0])
    fill_value = np.array(7.0)
    dtype = np.float32
    order = 'K'
    subok = True
    shape = (3,)

    input_dict = {
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "order": order,
        "subok": subok,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    a = np.array(1)
    fill_value = np.array(1)
    dtype = np.int32
    order = 'K'
    subok = True
    shape = None

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
generated_inputs["tf.experimental.numpy.full_like"] = tf_experimental_numpy_full_like_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.full_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.full_like'.")

check_valid('tf.experimental.numpy.full_like', generated_inputs['tf.experimental.numpy.full_like'], lib="tf", suffix=0)
