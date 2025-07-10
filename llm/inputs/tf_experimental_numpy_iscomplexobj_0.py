
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_iscomplexobj_inputs():
    list_of_inputs = []
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Input 1: Complex scalar
    x = np.complex64(1 + 1j)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Real scalar
    x = np.float32(1.0)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex 1D array
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Real 1D array
    x = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex 2D array
    x = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Real 2D array
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex higher dimensional array
    x = np.array([[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]]], dtype=np.complex64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Real higher dimensional array
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex array with only zero imaginary parts
    x = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty array
    x = np.array([], dtype=np.complex64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Empty array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: int array
    x = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.iscomplexobj"] = tf_experimental_numpy_iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.iscomplexobj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.iscomplexobj'.")

check_valid('tf.experimental.numpy.iscomplexobj', generated_inputs['tf.experimental.numpy.iscomplexobj'], lib="tf", suffix=0)
