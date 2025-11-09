
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []

    val = (np.int32(1), np.int32(-2), np.int32(3))
    dtype = np.int32
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.array([1.0, -2.5], dtype=np.float64), np.array([3.3, 4.4], dtype=np.float64))
    dtype = np.float64
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.bool_(True), np.bool_(False), np.bool_(True), np.bool_(False))
    dtype = np.bool_
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.complex64(1+2j), np.complex64(-3+0.5j))
    dtype = np.complex64
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = ((np.int16(1), np.int16(2), np.int16(3)), (np.int16(-4), np.int16(-5), np.int16(-6)))
    dtype = np.int16
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32), np.full((2, 2), 7, dtype=np.float32))
    dtype = np.float32
    copy_flag = True
    ndmin = 3
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.uint8(255), np.uint8(0), np.uint8(128))
    dtype = np.uint8
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.arange(8, dtype=np.int64).reshape(2, 2, 2), np.arange(8, 16, dtype=np.int64).reshape(2, 2, 2))
    dtype = np.int64
    copy_flag = True
    ndmin = 4
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.array([-1.5, 0.0, 2.5], dtype=np.float16),)
    dtype = np.float16
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.int8(-5), np.float32(3.5), np.int8(10))
    dtype = np.float32
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = ((np.bool_(True), np.bool_(True)), (np.bool_(False), np.bool_(True)))
    dtype = np.bool_
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.array([1+0j, 0+2j], dtype=np.complex128), np.array([-3+4j, 5-6j], dtype=np.complex128))
    dtype = np.complex128
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array_3"] = tf_experimental_numpy_array_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.array_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.array_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.array', generated_inputs['tf.experimental.numpy.array_3'], lib="tf", suffix=3)
