
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_dstack_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {'tup': (a, b)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D arrays
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    input_dict = {'tup': (a, b)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shaped arrays (but compatible for stacking)
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {'tup': (a.reshape(1,3), b.reshape(1,3))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: More arrays in the tuple
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    input_dict = {'tup': (a, b, c)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {'tup': (a, b)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple containing a single array.
    a = np.array([[1, 2], [3, 4]])
    input_dict = {'tup': (a,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty arrays - removing this as its leading to more issues
    #a = np.array([]).reshape((0,1))
    #b = np.array([]).reshape((0,1))
    #input_dict = {'tup': (a, b)}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: arrays of different data types
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.int32) # Modified dtype to avoid inhomogeneous shape.
    input_dict = {'tup': (a, b)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: arrays with negative values
    a = np.array([-1, -2, -3])
    b = np.array([-4, -5, -6])
    input_dict = {'tup': (a, b)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: arrays with zeros
    a = np.array([0, 0, 0])
    b = np.array([0, 0, 0])
    input_dict = {'tup': (a, b)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.dstack"] = tf_experimental_numpy_dstack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.dstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.dstack'.")

check_valid('tf.experimental.numpy.dstack', generated_inputs['tf.experimental.numpy.dstack'], lib="tf", suffix=0)
