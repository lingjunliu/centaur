
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_nanprod_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([1.0, 2.0, np.nan, 4.0])
    axis = 0
    dtype = np.float32
    keepdims = False
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1.0, 2.0], [np.nan, 4.0]])
    axis = 0
    dtype = np.float64
    keepdims = True
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1.0, 2.0], [np.nan, 4.0]])
    axis = 1
    dtype = np.int32
    keepdims = False
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    axis = 2
    dtype = np.float32
    keepdims = True
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    axis = 0
    dtype = np.float64
    keepdims = False
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([1, 2, np.nan, 4], dtype=np.float32)
    axis = None
    dtype = np.int64
    keepdims = True
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    axis = 0
    dtype = np.float32
    keepdims = False
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    a = np.array([1.0, 2.0, np.nan, 4.0])
    axis = -1
    dtype = np.float32
    keepdims = False
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[1.0, 2.0], [np.nan, 4.0]])
    axis = -2
    dtype = np.int64
    keepdims = True
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    axis = -3
    dtype = np.float64
    keepdims = False
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.nanprod"] = tf_experimental_numpy_nanprod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.nanprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.nanprod'.")

check_valid('tf.experimental.numpy.nanprod', generated_inputs['tf.experimental.numpy.nanprod'], lib="tf", suffix=0)
