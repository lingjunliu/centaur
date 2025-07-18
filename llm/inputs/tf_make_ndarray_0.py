
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_make_ndarray_inputs():
    """
    Generates a list of valid inputs for the tf.make_ndarray function.
    The inputs are provided in numpy format as requested by the user's framework,
    which expects objects with a .shape attribute.
    """
    list_of_inputs = []

    # Input 1: Simple 2D int32 tensor
    tensor_1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict_1 = {'tensor': tensor_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D float32 tensor with negative values
    tensor_2 = np.array([-1.1, 0.0, 2.2, -3.3], dtype=np.float32)
    input_dict_2 = {'tensor': tensor_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D float64 tensor
    tensor_3 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict_3 = {'tensor': tensor_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 0D tensor (scalar) of type int64
    tensor_4 = np.array(42, dtype=np.int64)
    input_dict_4 = {'tensor': tensor_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Boolean tensor
    tensor_5 = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict_5 = {'tensor': tensor_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty tensor (shape [0])
    tensor_6 = np.array([], dtype=np.float32)
    input_dict_6 = {'tensor': tensor_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty tensor with non-zero dimensions (shape [2, 0, 3])
    tensor_7 = np.empty(shape=(2, 0, 3), dtype=np.int16)
    input_dict_7 = {'tensor': tensor_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Complex64 tensor
    tensor_8 = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict_8 = {'tensor': tensor_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Unsigned integer tensor (uint8)
    tensor_9 = np.array([[0, 255], [128, 1]], dtype=np.uint8)
    input_dict_9 = {'tensor': tensor_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: String tensor (byte strings) - numpy handles this with dtype=object
    tensor_10 = np.array([b"hello", b"world"], dtype=object)
    input_dict_10 = {'tensor': tensor_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Float tensor with special values (inf, -inf, nan)
    tensor_11 = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
    input_dict_11 = {'tensor': tensor_11}
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Complex128 tensor
    tensor_12 = np.array([[1.1 + 2.2j], [3.3 - 4.4j]], dtype=np.complex128)
    input_dict_12 = {'tensor': tensor_12}
    list_of_inputs.append(copy.deepcopy(input_dict_12))

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
