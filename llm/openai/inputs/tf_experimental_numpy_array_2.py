
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_array_2_inputs():
    list_of_inputs = []

    # Input 1
    val = [1, 2, 3]
    dtype = np.int32
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    val = [[-1, 0, 7]]
    dtype = np.int64
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    val = [1.5, -2.5, 3.0]
    dtype = np.float32
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    val = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    dtype = np.int8
    copy_flag = False
    ndmin = 3
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    val = [True, False, True, False]
    dtype = np.bool_
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    val = [0, 1, 0, 2]
    dtype = np.bool_
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    val = [[1000, -2000, 3000], [4000, -5000, 6000]]
    dtype = np.int32
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    val = [[1.2, 3.4], [5.6, 7.8]]
    dtype = np.dtype('float64')
    copy_flag = False
    ndmin = 4
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    val = [0, 255, 128]
    dtype = np.uint8
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    val = [[0.0]]
    dtype = np.float16
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    val = [[1, 2], [3, 4], [5, 6]]
    dtype = np.int64
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    val = [1.0, 2.0, 3.0, 4.0]
    dtype = np.float64
    copy_flag = True
    ndmin = 3
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array_2"] = tf_experimental_numpy_array_2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.array_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.array_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.array', generated_inputs['tf.experimental.numpy.array_2'], lib="tf", suffix=2)
