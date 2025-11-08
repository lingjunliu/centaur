
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_l2_loss_inputs():
    list_of_inputs = []

    t = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    name = "basic_float32_1d"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = (np.arange(12, dtype=np.float64).reshape(3, 4) - 5.5)
    name = "float64_2d_arange_shifted"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([[[1, -1], [2, -2]], [[3, -3], [4, -4]]], dtype=np.float16)
    name = "float16_3d_mixed"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array(3.14159265, dtype=np.float32)
    name = "float32_scalar"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.zeros((2, 3, 4, 1), dtype=np.float32)
    name = "float32_4d_zeros"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([], dtype=np.float32)
    name = "float32_empty"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    t = base[::2, 1::2, ::2]
    name = "float64_3d_strided_view"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([1e20, -1e20, 3e19], dtype=np.float32)
    name = "float32_large_magnitudes"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    if hasattr(np, "bfloat16"):
        t = np.linspace(-1.0, 1.0, 10, dtype=np.float32).astype(np.bfloat16)
        name = "bfloat16_1d_linspace"
    else:
        t = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
        name = "float32_1d_linspace_fallback"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = -3 * np.ones((5, 5), dtype=np.float16)
    name = "float16_2d_all_negative"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    name = "float64_with_nan_inf"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.ones((1, 2, 1, 2, 3), dtype=np.float32)
    name = "float32_5d_ones"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([-7.0], dtype=np.float32)
    name = "float32_single_element"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.l2_loss"] = tf_nn_l2_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.l2_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.l2_loss'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.l2_loss', generated_inputs['tf.nn.l2_loss'], lib="tf", suffix=0)
