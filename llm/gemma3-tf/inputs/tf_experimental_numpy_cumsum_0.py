
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_cumsum_inputs():
    list_of_inputs = []

    a = np.array([1, 2, 3])
    axis = None
    dtype = np.int32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]])
    axis = 0
    dtype = np.int64
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]])
    axis = 1
    dtype = np.float32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 2
    dtype = np.int16
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([-1, -2, -3])
    axis = None
    dtype = np.int32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.5, 2.5, 3.5])
    axis = None
    dtype = np.float64
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2], [3, 4]])
    axis = 0
    dtype = np.float32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    axis = 1
    dtype = np.int64
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1, 2, 3, 4, 5])
    axis = None
    dtype = np.int8
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = None
    dtype = np.int32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.cumsum"] = tf_experimental_numpy_cumsum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.cumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.cumsum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.cumsum', generated_inputs['tf.experimental.numpy.cumsum'], lib="tf", suffix=0)
