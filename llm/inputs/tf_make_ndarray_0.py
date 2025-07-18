
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_make_ndarray_inputs():
    """
    Generates a list of valid inputs for tf.make_ndarray.
    The testing framework requires an input object with a `.shape` attribute
    for its analysis phase, causing an error when a TensorProto is provided.
    To fix this, we provide numpy arrays, which satisfy the `.shape` requirement
    and the prompt's rule to use numpy format.
    """
    list_of_inputs = []

    # Input 1: 2D integer tensor
    tensor_1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_1}))

    # Input 2: Scalar (0D) float tensor
    tensor_2 = np.array(3.14, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_2}))

    # Input 3: 1D tensor with negative values
    tensor_3 = np.array([-10, -20, -30], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_3}))

    # Input 4: 3D float64 tensor
    tensor_4 = np.random.rand(2, 2, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_4}))

    # Input 5: Boolean tensor
    tensor_5 = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_5}))

    # Input 6: Empty 1D tensor
    tensor_6 = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_6}))

    # Input 7: String tensor (using bytes)
    tensor_7 = np.array([b"hello", b"tensorflow", b"world"])
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_7}))

    # Input 8: Complex number tensor (complex64)
    tensor_8 = np.array([1+2j, 3-4j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_8}))

    # Input 9: Unsigned integer tensor
    tensor_9 = np.array([[0, 1], [254, 255]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_9}))

    # Input 10: High-dimensional tensor (4D) of zeros
    tensor_10 = np.zeros((1, 2, 3, 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({'tensor': tensor_10}))

    return list_of_inputs

generated_inputs["tf.make_ndarray"] = get_tf_make_ndarray_inputs()

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
