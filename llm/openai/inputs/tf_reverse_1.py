
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []

    tensor = np.array([1, 2, 3, 4], dtype=np.int32)
    axis = np.array([0], dtype=np.int32)
    name = "rev_int32_1d_axis0"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    axis = np.array([-1], dtype=np.int64)
    name = "rev_float32_2d_last"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    axis = np.array([0, 2], dtype=np.int32)
    name = "rev_int64_3d_axes0_2"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([[[True, False], [False, True]]], dtype=bool)
    axis = np.array([-2], dtype=np.int64)
    name = "rev_bool_3d_axis_minus2"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = (np.arange(6).reshape(2, 3).astype(np.float32) + 1j * np.arange(6).reshape(2, 3).astype(np.float32)).astype(np.complex64)
    axis = np.array([], dtype=np.int32)
    name = "rev_complex64_2d_noaxis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    axis = np.array([1], dtype=np.int32)
    name = "rev_uint8_3d_axis1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.float16).reshape(1, 1, 2, 2, 1, 3)
    axis = np.array([5], dtype=np.int64)
    name = "rev_float16_6d_axis5"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(8, dtype=np.int8).reshape(1, 1, 1, 2, 2, 1, 2)
    axis = np.array([0, -3, 6], dtype=np.int64)
    name = "rev_int8_7d_axes0_minus3_6"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    shape = (1, 1, 1, 2, 1, 2, 1, 2)
    tensor = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    axis = np.array([-8, 2, -1], dtype=np.int64)
    name = "rev_float64_8d_axes_minus8_2_minus1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    axis = np.array([1], dtype=np.int32)
    name = "rev_int16_3d_axis1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    return list_of_inputs

generated_inputs["tf.reverse_1"] = tf_reverse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.reverse_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.reverse', generated_inputs['tf.reverse_1'], lib="tf", suffix=1)
