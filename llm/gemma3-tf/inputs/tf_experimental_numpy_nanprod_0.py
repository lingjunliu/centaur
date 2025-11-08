
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_nanprod_inputs():
    list_of_inputs = []

    a = np.array([1.0, 2.0, 3.0])
    axis = None
    dtype = np.float32
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    axis = 0
    dtype = np.float64
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    axis = 1
    dtype = np.float32
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    axis = 2
    dtype = np.float64
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0, 2.0, 3.0, np.nan])
    axis = None
    dtype = np.float16
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    axis = 0
    dtype = np.float32
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, np.nan]]])
    axis = 1
    dtype = np.float64
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0, 2.0, 3.0])
    axis = None
    dtype = np.float32
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    axis = 0
    dtype = np.float64
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    axis = None
    dtype = np.float16
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.nanprod"] = tf_experimental_numpy_nanprod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.nanprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.nanprod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.nanprod', generated_inputs['tf.experimental.numpy.nanprod'], lib="tf", suffix=0)
