
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isrealobj_inputs():
    list_of_inputs = []

    # Input 1: Real numpy array
    x = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex numpy array
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed numpy array (real and complex)
    x = np.array([1, 2+2j, 3], dtype=object)
    x = [tf.constant(1.0), tf.complex(2.0, 2.0), tf.constant(3.0)]
    input_dict = {"x": tf.stack(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D Real numpy array
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D Complex numpy array
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D Real numpy array
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D Complex numpy array
    x = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Real scalar
    x = np.array(5, dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex scalar
    x = np.array(5 + 2j, dtype=np.complex64)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Real array with negative values
    x = np.array([-1, -2, -3], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": tf.constant(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.isrealobj"] = tf_experimental_numpy_isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isrealobj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isrealobj'.")

check_valid('tf.experimental.numpy.isrealobj', generated_inputs['tf.experimental.numpy.isrealobj'], lib="tf", suffix=0)
