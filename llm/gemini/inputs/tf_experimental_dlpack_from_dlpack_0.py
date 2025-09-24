
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import ctypes

def tf_experimental_dlpack_from_dlpack_inputs():
    list_of_inputs = []

    # Helper function to create a dlpack capsule from a numpy array
    def numpy_to_dlpack_capsule(array):
        try:
            import dlpack
            tensor = dlpack.from_numpy(array)
            capsule = tensor.to_dlpack()
            capsule_str = str(capsule)
            return capsule_str
        except ImportError:
            return None

    # Input 1: Basic 1D array
    a1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    dlcapsule1 = numpy_to_dlpack_capsule(a1)
    if dlcapsule1:
        input_dict1 = {"dlcapsule": dlcapsule1}
        list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D array with float values
    a2 = np.array([[1.0, 2.5], [3.2, 4.8]], dtype=np.float32)
    dlcapsule2 = numpy_to_dlpack_capsule(a2)
    if dlcapsule2:
        input_dict2 = {"dlcapsule": dlcapsule2}
        list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D array
    a3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dlcapsule3 = numpy_to_dlpack_capsule(a3)
    if dlcapsule3:
        input_dict3 = {"dlcapsule": dlcapsule3}
        list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Array with boolean values
    a4 = np.array([[True, False], [False, True]], dtype=np.bool_)
    dlcapsule4 = numpy_to_dlpack_capsule(a4)
    if dlcapsule4:
        input_dict4 = {"dlcapsule": dlcapsule4}
        list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Empty array
    a5 = np.array([], dtype=np.int32)
    dlcapsule5 = numpy_to_dlpack_capsule(a5)
    if dlcapsule5:
        input_dict5 = {"dlcapsule": dlcapsule5}
        list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Array with negative values
    a6 = np.array([-1, -2, -3], dtype=np.int32)
    dlcapsule6 = numpy_to_dlpack_capsule(a6)
    if dlcapsule6:
        input_dict6 = {"dlcapsule": dlcapsule6}
        list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Large array
    a7 = np.random.rand(100, 100).astype(np.float32)
    dlcapsule7 = numpy_to_dlpack_capsule(a7)
    if dlcapsule7:
        input_dict7 = {"dlcapsule": dlcapsule7}
        list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Array with int64 type
    a8 = np.array([1, 2, 3], dtype=np.int64)
    dlcapsule8 = numpy_to_dlpack_capsule(a8)
    if dlcapsule8:
        input_dict8 = {"dlcapsule": dlcapsule8}
        list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Array with uint8 type
    a9 = np.array([1, 2, 3], dtype=np.uint8)
    dlcapsule9 = numpy_to_dlpack_capsule(a9)
    if dlcapsule9:
        input_dict9 = {"dlcapsule": dlcapsule9}
        list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Array with different shape (1, 5, 1)
    a10 = np.array([[[1], [2], [3], [4], [5]]], dtype=np.int32)
    dlcapsule10 = numpy_to_dlpack_capsule(a10)
    if dlcapsule10:
        input_dict10 = {"dlcapsule": dlcapsule10}
        list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Array with zero values
    a11 = np.array([0, 0, 0], dtype=np.int32)
    dlcapsule11 = numpy_to_dlpack_capsule(a11)
    if dlcapsule11:
        input_dict11 = {"dlcapsule": dlcapsule11}
        list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs["tf.experimental.dlpack.from_dlpack"] = tf_experimental_dlpack_from_dlpack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.dlpack.from_dlpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.dlpack.from_dlpack'.")

check_valid('tf.experimental.dlpack.from_dlpack', generated_inputs['tf.experimental.dlpack.from_dlpack'], lib="tf", suffix=0)
