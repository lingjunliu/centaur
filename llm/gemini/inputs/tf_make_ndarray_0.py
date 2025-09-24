
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_make_ndarray_inputs():
    """
    Generates a list of valid inputs for the tf.make_ndarray function.
    The user's test harness expects a numpy-like object with a .shape attribute,
    so we provide numpy arrays directly. The harness is expected to handle the
    conversion to a TensorProto before calling the API.
    """
    list_of_inputs = []

    # Input 1: Basic 2D int32 array
    list_of_inputs.append(copy.deepcopy({'tensor': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)}))
    
    # Input 2: 1D float32 array with negative values
    list_of_inputs.append(copy.deepcopy({'tensor': np.array([-1.1, 0.0, 2.2, -3.3], dtype=np.float32)}))

    # Input 3: 3D float64 array
    list_of_inputs.append(copy.deepcopy({'tensor': np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float64)}))

    # Input 4: 0D (scalar) int64 array
    list_of_inputs.append(copy.deepcopy({'tensor': np.array(987654321098765432, dtype=np.int64)}))

    # Input 5: Empty 1D array
    list_of_inputs.append(copy.deepcopy({'tensor': np.array([], dtype=np.float32)}))

    # Input 6: Array with a zero dimension
    list_of_inputs.append(copy.deepcopy({'tensor': np.zeros(shape=(2, 0, 3), dtype=np.int32)}))

    # Input 7: Boolean array
    list_of_inputs.append(copy.deepcopy({'tensor': np.array([[True, False], [False, True]], dtype=np.bool_)}))

    # Input 8: Complex64 array
    list_of_inputs.append(copy.deepcopy({'tensor': np.array([1 + 2j, 3 - 4j, -5 - 6j], dtype=np.complex64)}))

    # Input 9: Complex128 array
    list_of_inputs.append(copy.deepcopy({'tensor': np.array([[5.5 + 6.6j]], dtype=np.complex128)}))
    
    # Input 10: Unsigned integer array
    list_of_inputs.append(copy.deepcopy({'tensor': np.array([0, 255, 128], dtype=np.uint8)}))
    
    # Input 11: 4D array of float16
    list_of_inputs.append(copy.deepcopy({'tensor': np.ones((1, 2, 2, 1), dtype=np.float16)}))

    # Input 12: String array
    list_of_inputs.append(copy.deepcopy({'tensor': np.array([["hello", "world"], ["tensorflow", "rules"]], dtype=object)}))
    
    return list_of_inputs

generated_inputs["tf.make_ndarray"] = tf_make_ndarray_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.make_ndarray' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.make_ndarray'.")

check_valid('tf.make_ndarray', generated_inputs['tf.make_ndarray'], lib="tf", suffix=0)
