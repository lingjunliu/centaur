
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import torch
import copy

def tf_raw_ops_Selu_inputs():
    list_of_inputs = []

    features = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    name = "selu_case_1"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([[-2.5, 0.0, 2.5], [3.3, -4.4, 5.5]], dtype=np.float16)
    name = "selu_case_2"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.linspace(-3, 3, 24, dtype=np.float64).reshape(2, 3, 4)
    name = "selu_case_3"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array(-1.23, dtype=np.float32)
    name = "selu_case_4_scalar"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = (((np.arange(24).reshape(2, 1, 3, 4) - 12) / 5.0)).astype(np.float32)
    name = "selu_case_5_4d"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([20.0, -20.0, 10.0, -10.0], dtype=np.float32)
    name = "selu_case_6_extremes"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([-1e-8, 0.0, 1e-8], dtype=np.float64)
    name = "selu_case_7_small_vals"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0], dtype=np.float32)
    name = "selu_case_8_specials"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([], dtype=np.float32)
    name = "selu_case_9_empty"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    features = base[:, ::2]
    name = "selu_case_10_strided"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = (np.arange(12, dtype=np.float16).reshape(1, 2, 1, 2, 3) - 6) / np.float16(3.0)
    name = "selu_case_11_5d_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    mat = (np.arange(16, dtype=np.float64).reshape(4, 4) - 8.0) / 4.0
    features = mat.T
    name = "selu_case_12_transposed_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Selu"] = tf_raw_ops_Selu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Selu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Selu', generated_inputs['tf.raw_ops.Selu'], lib="tf", suffix=0)
