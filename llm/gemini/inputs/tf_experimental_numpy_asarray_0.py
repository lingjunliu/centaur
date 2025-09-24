
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_asarray_inputs():
    list_of_inputs = []

    # Input 1: Basic numpy array conversion
    a = np.array([1, 2, 3])
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Numpy array with specified dtype
    a = np.array([1.0, 2.0, 3.0])
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multidimensional numpy array
    a = np.array([[1, 2], [3, 4]])
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numpy array with different dtype
    a = np.array([1, 2, 3], dtype=np.int64)
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Numpy array with negative values
    a = np.array([-1, 0, 1])
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Numpy array with boolean values
    a = np.array([True, False, True])
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Numpy array with complex values
    a = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D numpy array
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero-dimensional numpy array
    a = np.array(5)
    dtype = None
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Numpy array with explicit numpy.float64
    a = np.array([1, 2, 3], dtype=np.float64)
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.asarray"] = tf_experimental_numpy_asarray_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.asarray' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.asarray'.")

check_valid('tf.experimental.numpy.asarray', generated_inputs['tf.experimental.numpy.asarray'], lib="tf", suffix=0)
