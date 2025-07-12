
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import ctypes
from dlpack import to_dlpack

def tf_experimental_dlpack_from_dlpack_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array
    np_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with float64
    np_array = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with complex64
    np_array = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]]], dtype=np.complex64)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: array with bool
    np_array = np.array([[True, False], [False, True]], dtype=np.bool_)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty array
    np_array = np.array([], dtype=np.int32)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with negative values
    np_array = np.array([-1, -2, -3], dtype=np.int32)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger array
    np_array = np.random.rand(100, 100).astype(np.float32)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with only zeros
    np_array = np.zeros((5, 5), dtype=np.float32)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with uint8
    np_array = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with int64
    np_array = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    dlcapsule = to_dlpack(np_array)
    dlcapsule_str = str(dlcapsule)
    input_dict = {"dlcapsule": dlcapsule_str}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.dlpack.from_dlpack"] = tf_experimental_dlpack_from_dlpack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.dlpack.from_dlpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.dlpack.from_dlpack'.")

check_valid('tf.experimental.dlpack.from_dlpack', generated_inputs['tf.experimental.dlpack.from_dlpack'], lib="tf", suffix=0)
