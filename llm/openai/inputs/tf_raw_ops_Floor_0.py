
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floor_inputs():
    list_of_inputs = []

    x = np.array(3.7, dtype=np.float32)
    name = "floor_case_1"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-2.3, dtype=np.float64)
    name = "floor_case_2"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, -0.0, 1.999, -1.001, 1e10], dtype=np.float64)
    name = "floor_case_3"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.1, 0.9], [-0.1, -0.9]], dtype=np.float16)
    name = "floor_case_4"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(-5, 5, num=24, dtype=np.float32).reshape(2, 3, 4) + 0.499
    name = "floor_case_5"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "floor_case_6"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, np.pi, -np.pi, 2.999, -2.001, 100.999], dtype=np.float64).reshape(1, 2, 1, 3)
    name = "floor_case_8_4d"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.inf, -np.inf, np.nan, 5.5, -5.5], dtype=np.float32)
    name = "floor_case_9_inf_nan"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = (np.arange(20, dtype=np.float32) / 3.0) - 3.0
    x = base[::2]
    name = "floor_case_10_noncontig"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.asfortranarray(np.array([[1.2, -3.4, 5.6], [7.8, -9.0, 0.0]], dtype=np.float32))
    name = "floor_case_11_fortran_order"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1e308, 1e-308, -1e-308, 1e308], dtype=np.float64)
    name = "floor_case_12_extremes"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Floor"] = tf_raw_ops_floor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Floor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Floor', generated_inputs['tf.raw_ops.Floor'], lib="tf", suffix=0)
