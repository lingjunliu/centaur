
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_stack_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D arrays, axis=0
    arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, axis=0
    arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, axis=1
    arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = 1
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, axis=-1
    arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = -1
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays, axis=0
    arrays = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D arrays with negative values
    arrays = [np.array([-1, -2, -3]), np.array([-4, -5, -6])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D arrays, axis=-2
    arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = -2
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single array
    arrays = [np.array([1, 2, 3])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays, axis=1
    arrays = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = 1
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D arrays, axis = 0, different types
    arrays = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: With floats
    arrays = [np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Complex numbers
    arrays = [np.array([1+1j, 2+2j, 3+3j]), np.array([4+4j, 5+5j, 6+6j])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: Bool
    arrays = [np.array([True, False, True]), np.array([False, True, False])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: Mixed types (should error)
    arrays = [np.array([1, 2, 3]), np.array([1.1, 2.2, 3.3])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.stack"] = tf_experimental_numpy_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.stack'.")

check_valid('tf.experimental.numpy.stack', generated_inputs['tf.experimental.numpy.stack'], lib="tf", suffix=0)
