
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_reverse_inputs():
    list_of_inputs = []

    tensor = np.arange(10, dtype=np.int32)
    axis = (0,)
    name = "rev_case_1_int32_1d"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    axis = (-1,)
    name = "rev_case_2_float32_2d_last_axis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    axis = (1, 2)
    name = "rev_case_3_int64_3d_axes_1_2"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 4)
    axis = (3,)
    name = "rev_case_4_float64_4d_axis3"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array(42.5, dtype=np.float64)
    axis = ()
    name = "rev_case_5_scalar_noop"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = (np.arange(2 * 1 * 2 * 1 * 2) % 2 == 0).reshape(2, 1, 2, 1, 2)
    axis = (0, 3)
    name = "rev_case_6_bool_5d_axes_0_3"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(2 * 1 * 2 * 1 * 2 * 1, dtype=np.float32).reshape(2, 1, 2, 1, 2, 1)
    axis = (0, 2, 4)
    name = "rev_case_7_float32_6d_axes_0_2_4"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1 * 1 * 2 * 1 * 2 * 1 * 3, dtype=np.int32).reshape(1, 1, 2, 1, 2, 1, 3)
    axis = (-7, -1)
    name = "rev_case_8_int32_7d_neg_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(2, dtype=np.int64).reshape(1, 1, 1, 1, 1, 1, 1, 2)
    axis = (-1,)
    name = "rev_case_9_int64_8d_last_axis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = (np.arange(24) % 3 == 0).reshape(2, 3, 4)
    axis = (-3, -2, -1)
    name = "rev_case_10_bool_3d_all_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(20, dtype=np.float64).reshape(5, 4)
    axis = (0, 1)
    name = "rev_case_11_float64_2d_both_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    axis = (1,)
    name = "rev_case_12_float32_3d_axis1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    return list_of_inputs

generated_inputs["tf.reverse_3"] = tf_reverse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.reverse_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.reverse', generated_inputs['tf.reverse_3'], lib="tf", suffix=3)
