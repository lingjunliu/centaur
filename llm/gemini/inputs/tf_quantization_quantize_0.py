
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_quantize_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    min_range_tensor = np.array(-2.0, dtype=np.float32)
    max_range_tensor = np.array(2.0, dtype=np.float32)
    T_dtype = tf.qint8
    mode_str = "MIN_COMBINED"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_1"
    narrow_range_bool = False
    axis_int = None
    ensure_minimum_range_float = 0.01

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    min_range_tensor = np.array(-5.0, dtype=np.float32)
    max_range_tensor = np.array(5.0, dtype=np.float32)
    T_dtype = tf.quint8
    mode_str = "MIN_FIRST"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_2"
    narrow_range_bool = True
    axis_int = None
    ensure_minimum_range_float = 0.0

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    min_range_tensor = np.array(-1.0, dtype=np.float32)
    max_range_tensor = np.array(4.0, dtype=np.float32)
    T_dtype = tf.qint32
    mode_str = "SCALED"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_3"
    narrow_range_bool = False
    axis_int = None
    ensure_minimum_range_float = 0.01

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Axis quantization
    input_tensor = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    min_range_tensor = np.array([-1.0, 2.0], dtype=np.float32)
    max_range_tensor = np.array([1.0, 4.0], dtype=np.float32)
    T_dtype = tf.qint8
    mode_str = "MIN_COMBINED"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_4"
    narrow_range_bool = False
    axis_int = 0
    ensure_minimum_range_float = 0.01

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Ensure Minimum Range
    input_tensor = np.array([0.0, 0.005, 0.01, 0.015], dtype=np.float32)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(0.01, dtype=np.float32)
    T_dtype = tf.quint8
    mode_str = "MIN_COMBINED"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_5"
    narrow_range_bool = False
    axis_int = None
    ensure_minimum_range_float = 0.1

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative min/max ranges
    input_tensor = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    min_range_tensor = np.array(-10.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    T_dtype = tf.qint16
    mode_str = "MIN_FIRST"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_6"
    narrow_range_bool = False
    axis_int = None
    ensure_minimum_range_float = 0.01

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D Tensor
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(1.0, dtype=np.float32)
    T_dtype = tf.quint16
    mode_str = "SCALED"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_7"
    narrow_range_bool = True
    axis_int = None
    ensure_minimum_range_float = 0.0

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: qint32
    input_tensor = np.array([-1e5, 0, 1e5], dtype=np.float32)
    min_range_tensor = np.array(-1e5, dtype=np.float32)
    max_range_tensor = np.array(1e5, dtype=np.float32)
    T_dtype = tf.qint32
    mode_str = "MIN_COMBINED"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_8"
    narrow_range_bool = False
    axis_int = None
    ensure_minimum_range_float = 0.01

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D Tensor with axis
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    num_channels = input_tensor.shape[1]
    min_range_tensor = np.random.rand(num_channels).astype(np.float32)
    max_range_tensor = np.random.rand(num_channels).astype(np.float32) + 1.0
    T_dtype = tf.qint8
    mode_str = "MIN_COMBINED"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_9"
    narrow_range_bool = False
    axis_int = 1
    ensure_minimum_range_float = 0.01

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger input values
    input_tensor = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0], dtype=np.float32)
    min_range_tensor = np.array(-1000.0, dtype=np.float32)
    max_range_tensor = np.array(1000.0, dtype=np.float32)
    T_dtype = tf.quint8
    mode_str = "SCALED"
    round_mode_str = "HALF_AWAY_FROM_ZERO"
    name_str = "quantize_example_10"
    narrow_range_bool = False
    axis_int = None
    ensure_minimum_range_float = 0.01

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "T": T_dtype,
        "mode": mode_str,
        "round_mode": round_mode_str,
        "name": name_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "ensure_minimum_range": ensure_minimum_range_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.quantization.quantize"] = tf_quantization_quantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.quantization.quantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.quantize'.")

check_valid('tf.quantization.quantize', generated_inputs['tf.quantization.quantize'], lib="tf", suffix=0)
