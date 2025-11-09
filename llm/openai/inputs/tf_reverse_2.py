
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_2_inputs():
    list_of_inputs = []

    tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = [0]
    name = "rev_1d_int32"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    axis = [1]
    name = "rev_2d_float32_axis1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([[[True, False, True],
                        [False, False, True]],
                       [[True, True, False],
                        [False, True, False]]], dtype=np.bool_)
    axis = [-1]
    name = "rev_3d_bool_last"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1*2*3*4, dtype=np.int32).reshape(1, 2, 3, 4)
    axis = [1, 3]
    name = "rev_4d_int32_multi"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1*1*2*2*3, dtype=np.float32).reshape(1, 1, 2, 2, 3)
    axis = [-3]
    name = "rev_5d_float32_neg"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array(42, dtype=np.int64)
    axis = []
    name = "rev_scalar_int64_empty_axis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.int64).reshape(2, 2, 3)
    axis = [2, 0]
    name = "rev_3d_int64_multi"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.empty((3, 0, 4), dtype=np.float64)
    axis = [1]
    name = "rev_empty_dim_float64"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([10, -3, 7, 0, 5, -8, 2, 9], dtype=np.int32)
    axis = [-1]
    name = "rev_1d_int32_negaxis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(2*3*4*1*2*1, dtype=np.int32).reshape(2, 3, 4, 1, 2, 1)
    axis = [0, 2, 5]
    name = "rev_6d_int32_multi"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1*1*1*2*1*2*1, dtype=np.int32).reshape(1, 1, 1, 2, 1, 2, 1)
    axis = [3, 5]
    name = "rev_7d_int32_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([[1.0, 2.0],
                       [3.0, 4.0]], dtype=np.float64)
    axis = []
    name = "rev_2d_float64_noop"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1*2*1*2*1*2*1*2, dtype=np.int64).reshape(1, 2, 1, 2, 1, 2, 1, 2)
    axis = [0, 2, 4, 6]
    name = "rev_8d_int64_even_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(2*3*1*5, dtype=np.float32).reshape(2, 3, 1, 5)
    axis = [-4, -1]
    name = "rev_4d_float32_negaxes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    return list_of_inputs

generated_inputs["tf.reverse_2"] = tf_reverse_2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.reverse_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.reverse', generated_inputs['tf.reverse_2'], lib="tf", suffix=2)
