
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_quantize_and_dequantize_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_min = np.array(-2.0, dtype=np.float32)
    input_max = np.array(2.0, dtype=np.float32)
    signed_input = True
    num_bits = 8
    range_given = True
    round_mode = "HALF_TO_EVEN"
    name = "test_quantize_dequantize_1"
    narrow_range = False
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    input_min = np.array(-1.0, dtype=np.float32)
    input_max = np.array(2.0, dtype=np.float32)
    signed_input = False
    num_bits = 4
    range_given = True
    round_mode = "HALF_UP"
    name = "test_quantize_dequantize_2"
    narrow_range = True
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(1.0, dtype=np.float32)
    signed_input = True
    num_bits = 8
    range_given = True
    round_mode = "HALF_TO_EVEN"
    name = "test_quantize_dequantize_3"
    narrow_range = False
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[-1.0, -0.5], [0.0, 0.5]], dtype=np.float32)
    input_min = np.array(-1.0, dtype=np.float32)
    input_max = np.array(1.0, dtype=np.float32)
    signed_input = True
    num_bits = 8
    range_given = True
    round_mode = "HALF_UP"
    name = "test_quantize_dequantize_4"
    narrow_range = True
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_min = np.array(1.0, dtype=np.float32)
    input_max = np.array(5.0, dtype=np.float32)
    signed_input = False
    num_bits = 8
    range_given = True
    round_mode = "HALF_TO_EVEN"
    name = "test_quantize_dequantize_5"
    narrow_range = False
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - AXIS needs scalar input_min and input_max
    input_tensor = np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=np.float32)
    input_min = np.array(-2.0, dtype=np.float32)
    input_max = np.array(1.0, dtype=np.float32)
    signed_input = True
    num_bits = 8
    range_given = True
    round_mode = "HALF_TO_EVEN"
    name = "test_quantize_dequantize_6"
    narrow_range = False
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_min = np.array(1.0, dtype=np.float32)
    input_max = np.array(8.0, dtype=np.float32)
    signed_input = False
    num_bits = 4
    range_given = True
    round_mode = "HALF_UP"
    name = "test_quantize_dequantize_7"
    narrow_range = True
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_min = np.array(1.0, dtype=np.float32)
    input_max = np.array(3.0, dtype=np.float32)
    signed_input = True
    num_bits = 2
    range_given = True
    round_mode = "HALF_TO_EVEN"
    name = "test_quantize_dequantize_8"
    narrow_range = False
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([-5.0, -4.0, -3.0, -2.0, -1.0], dtype=np.float32)
    input_min = np.array(-5.0, dtype=np.float32)
    input_max = np.array(-1.0, dtype=np.float32)
    signed_input = True
    num_bits = 6
    range_given = True
    round_mode = "HALF_UP"
    name = "test_quantize_dequantize_9"
    narrow_range = True
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_min = np.array(1.0, dtype=np.float32)
    input_max = np.array(8.0, dtype=np.float32)
    signed_input = False
    num_bits = 8
    range_given = True
    round_mode = "HALF_TO_EVEN"
    name = "test_quantize_dequantize_10"
    narrow_range = False
    axis = None

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "name": name,
        "narrow_range": narrow_range,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.quantization.quantize_and_dequantize"] = tf_quantization_quantize_and_dequantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.quantization.quantize_and_dequantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.quantize_and_dequantize'.")

check_valid('tf.quantization.quantize_and_dequantize', generated_inputs['tf.quantization.quantize_and_dequantize'], lib="tf", suffix=0)
