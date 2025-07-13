
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizeAndDequantizeV4_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_min_tensor = np.array(-2.0, dtype=np.float32)
    input_max_tensor = np.array(2.0, dtype=np.float32)
    signed_input = True
    num_bits = 8
    range_given = True
    round_mode = "HALF_TO_EVEN"
    narrow_range = False
    axis = -1
    name = "quantize_dequantize_1"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_min_tensor = np.array(-1.0, dtype=np.float32)
    input_max_tensor = np.array(1.0, dtype=np.float32)
    signed_input = False
    num_bits = 4
    range_given = True
    round_mode = "HALF_UP"
    narrow_range = True
    axis = -1
    name = "quantize_dequantize_2"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    input_min_tensor = np.array(-1.0, dtype=np.float64)
    input_max_tensor = np.array(2.0, dtype=np.float64)
    signed_input = True
    num_bits = 8
    range_given = True
    round_mode = "HALF_TO_EVEN"
    narrow_range = False
    axis = -1
    name = "quantize_dequantize_3"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    input_tensor = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    input_min_tensor = np.array(-2.0, dtype=np.float32)
    input_max_tensor = np.array(3.0, dtype=np.float32)
    signed_input = True
    num_bits = 7
    range_given = True
    round_mode = "HALF_UP"
    narrow_range = True
    axis = -1
    name = "quantize_dequantize_4"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_min_tensor = np.array(1.0, dtype=np.float32)
    input_max_tensor = np.array(8.0, dtype=np.float32)
    signed_input = False
    num_bits = 8
    range_given = True
    round_mode = "HALF_TO_EVEN"
    narrow_range = False
    axis = -1
    name = "quantize_dequantize_5"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(10.0, dtype=np.float32)
    signed_input = True
    num_bits = 2
    range_given = True
    round_mode = "HALF_UP"
    narrow_range = True
    axis = -1
    name = "quantize_dequantize_6"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_min_tensor = np.array(1.0, dtype=np.float32)
    input_max_tensor = np.array(5.0, dtype=np.float32)
    signed_input = True
    num_bits = 8
    range_given = False
    round_mode = "HALF_TO_EVEN"
    narrow_range = False
    axis = -1
    name = "quantize_dequantize_7"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(6.0, dtype=np.float32)
    signed_input = False
    num_bits = 6
    range_given = True
    round_mode = "HALF_UP"
    narrow_range = True
    axis = -1
    name = "quantize_dequantize_8"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, using float16 instead of bfloat16
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_min_tensor = np.array(1.0, dtype=np.float16)
    input_max_tensor = np.array(4.0, dtype=np.float16)
    signed_input = True
    num_bits = 8
    range_given = True
    round_mode = "HALF_TO_EVEN"
    narrow_range = False
    axis = -1
    name = "quantize_dequantize_9"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_min_tensor = np.array(0.0, dtype=np.float16)
    input_max_tensor = np.array(5.0, dtype=np.float16)
    signed_input = False
    num_bits = 4
    range_given = True
    round_mode = "HALF_UP"
    narrow_range = True
    axis = -1
    name = "quantize_dequantize_10"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = np.array(1.0, dtype=np.float64)
    input_min_tensor = np.array(0.0, dtype=np.float64)
    input_max_tensor = np.array(5.0, dtype=np.float64)
    signed_input = False
    num_bits = 4
    range_given = True
    round_mode = "HALF_UP"
    narrow_range = True
    axis = -1
    name = "quantize_dequantize_11"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "signed_input": signed_input,
        "num_bits": num_bits,
        "range_given": range_given,
        "round_mode": round_mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizeAndDequantizeV4"] = tf_raw_ops_QuantizeAndDequantizeV4_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizeAndDequantizeV4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeAndDequantizeV4'.")

check_valid('tf.raw_ops.QuantizeAndDequantizeV4', generated_inputs['tf.raw_ops.QuantizeAndDequantizeV4'], lib="tf", suffix=0)
