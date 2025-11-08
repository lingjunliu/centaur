
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_prod_inputs():
    list_of_inputs = []

    input_arr = np.array([1, 2, 3], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i1",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    axis = np.array(-1, dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i2",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.linspace(-1.5, 2.5, num=24, dtype=np.float64).reshape(2, 3, 4)
    axis = np.array([0, 2], dtype=np.int64)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i3",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([0, 5, 10], dtype=np.uint8)
    axis = np.array(0, dtype=np.int32)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i4",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[-2, 3], [4, -5]], dtype=np.int16)
    axis = np.array([0], dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i5",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[1], [2], [3]], [[-1], [0], [4]]], dtype=np.int8)
    axis = np.array([1], dtype=np.int64)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i6",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array(
        [[[1+2j, 3+4j], [5-1j, 2+0j]],
         [[0+1j, -1-1j], [2+2j, -3+0j]]],
        dtype=np.complex64
    )
    axis = np.array(-2, dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i7",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([2, 3, 4, 5], dtype=np.int64)
    axis = np.array(0, dtype=np.int64)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i8",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(-12, 12, dtype=np.float16).reshape(2, 3, 4)
    axis = np.array([1], dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i9",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    axis = np.array(-1, dtype=np.int64)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i10",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(1, 17, dtype=np.int32).reshape(2, 2, 2, 2)
    axis = np.array([-1, -2, -3, -4], dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i11",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[1+0j, 2-1j, 3+3j]], dtype=np.complex64)
    axis = np.array([0], dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i12",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[10, 20, 30], [2, 3, 4]], dtype=np.int64)
    axis = np.array(0, dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i13",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Prod"] = tf_raw_ops_prod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Prod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Prod', generated_inputs['tf.raw_ops.Prod'], lib="tf", suffix=0)
