
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []

    val = np.array([1, 2, 3], dtype=np.int32)
    dtype = np.int32
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([-5, 0, 7, -9], dtype=np.int64)
    dtype = np.int64
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([[1.5, -2.0], [3.3, 4.4]], dtype=np.float32)
    dtype = np.float32
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64)
    dtype = np.complex64
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([[True, False], [False, True]], dtype=np.bool_)
    dtype = np.bool_
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    dtype = np.float64
    copy_flag = False
    ndmin = 3
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([10, 20, 30], dtype=np.uint8)
    dtype = np.uint8
    copy_flag = False
    ndmin = 4
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array(7.25, dtype=np.float64)
    dtype = np.float64
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([], dtype=np.float32)
    dtype = np.float32
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([0, 255], dtype=np.uint8)
    dtype = np.uint16
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.zeros((2, 1, 3, 1), dtype=np.float32)
    dtype = np.float32
    copy_flag = False
    ndmin = 4
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([[np.nan, np.inf], [-np.inf, -1.0]], dtype=np.float64)
    dtype = np.float64
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.arange(100, dtype=np.int64)[::3]
    dtype = np.int64
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([1.2, -3.4, 5.6], dtype=np.float32)
    dtype = np.int32
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array(True, dtype=np.bool_)
    dtype = np.bool_
    copy_flag = False
    ndmin = 5
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array_1"] = tf_experimental_numpy_array_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.array_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.array_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.array', generated_inputs['tf.experimental.numpy.array_1'], lib="tf", suffix=1)
