
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_isnan_inputs():
    list_of_inputs = []

    x = np.array([5.0, np.nan, -3.2, np.inf, -np.inf, 0.0], dtype=np.float32)
    name = "isnan_case_1"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[np.nan, 1.0], [2.5, -np.inf], [np.inf, 0.0]], dtype=np.float64)
    name = "isnan_case_2"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[0.0, np.nan, 3.0], [4.0, 5.0, np.nan]], [[-1.0, -2.0, -3.0], [np.inf, -np.inf, 7.0]]], dtype=np.float16)
    name = "isnan_case_3"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array(np.nan, dtype=np.float32)
    name = "isnan_case_4_scalar_nan"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    name = "isnan_case_5_4d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([], dtype=np.float64)
    name = "isnan_case_6_empty_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    base = np.arange(20, dtype=np.float32)
    base[3] = np.nan
    base[10] = np.nan
    x = base[::2]
    name = "isnan_case_7_noncontig"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-0.0, 0.0, np.nan, 1e308, -1e308, np.nan], dtype=np.float64)
    name = "isnan_case_8_large_vals"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[np.inf, -np.inf, np.nan], [1.0, 2.0, -3.0]], dtype=np.float16)
    name = "isnan_case_9_f16_mixed"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.zeros((2, 0, 3), dtype=np.float32)
    name = "isnan_case_10_empty_3d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([np.float32(1e-45), np.float32(0.0), np.nan, np.float32(-1e-45)], dtype=np.float32)
    name = "isnan_case_11_subnormals"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[[np.nan, 1.0, 2.0]], [[3.0, np.nan, 4.0]]]]], dtype=np.float32)
    name = "isnan_case_12_5d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.IsNan"] = tf_raw_ops_isnan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.IsNan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsNan'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.IsNan', generated_inputs['tf.raw_ops.IsNan'], lib="tf", suffix=0)
