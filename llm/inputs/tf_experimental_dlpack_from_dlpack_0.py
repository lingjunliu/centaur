
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import ctypes
import pyarrow as pa
from pyarrow import dlpack

def tf_experimental_dlpack_from_dlpack_inputs():
    list_of_inputs = []

    # Helper function to create a DLPack capsule from a numpy array
    def numpy_to_dlpack_capsule(np_array):
        arrow_array = pa.array(np_array)
        dlcapsule = dlpack.to_dlpack(arrow_array)
        dlcapsule_str = str(dlcapsule)

        return dlcapsule_str

    # Input 1: Simple 1D array
    np_array_1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    capsule_1 = numpy_to_dlpack_capsule(np_array_1)
    input_dict_1 = {"dlcapsule": capsule_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D array with float32
    np_array_2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    capsule_2 = numpy_to_dlpack_capsule(np_array_2)
    input_dict_2 = {"dlcapsule": capsule_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D array with complex64
    np_array_3 = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    capsule_3 = numpy_to_dlpack_capsule(np_array_3)
    input_dict_3 = {"dlcapsule": capsule_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Array with bool type
    np_array_4 = np.array([True, False, True], dtype=np.bool_)
    capsule_4 = numpy_to_dlpack_capsule(np_array_4)
    input_dict_4 = {"dlcapsule": capsule_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Array with int64 type
    np_array_5 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    capsule_5 = numpy_to_dlpack_capsule(np_array_5)
    input_dict_5 = {"dlcapsule": capsule_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty array
    np_array_6 = np.array([], dtype=np.float32)
    arrow_array_6 = pa.array(np_array_6)
    dlcapsule_6 = dlpack.to_dlpack(arrow_array_6)
    dlcapsule_str_6 = str(dlcapsule_6)
    input_dict_6 = {"dlcapsule": dlcapsule_str_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Array with negative values and float64 type
    np_array_7 = np.array([-1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    capsule_7 = numpy_to_dlpack_capsule(np_array_7)
    input_dict_7 = {"dlcapsule": capsule_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Array with uint8 type
    np_array_8 = np.array([1, 2, 255], dtype=np.uint8)
    capsule_8 = numpy_to_dlpack_capsule(np_array_8)
    input_dict_8 = {"dlcapsule": capsule_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Array with large shape and int32
    np_array_9 = np.zeros((10, 10, 10), dtype=np.int32)
    capsule_9 = numpy_to_dlpack_capsule(np_array_9)
    input_dict_9 = {"dlcapsule": capsule_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Array with all zeros and float16
    np_array_10 = np.zeros((5, 5), dtype=np.float16)
    capsule_10 = numpy_to_dlpack_capsule(np_array_10)
    input_dict_10 = {"dlcapsule": capsule_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Array with non-contiguous data (transposed)
    np_array_11 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32).T
    capsule_11 = numpy_to_dlpack_capsule(np_array_11)
    input_dict_11 = {"dlcapsule": capsule_11}
    list_of_inputs.append(copy.deepcopy(input_dict_11))

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
