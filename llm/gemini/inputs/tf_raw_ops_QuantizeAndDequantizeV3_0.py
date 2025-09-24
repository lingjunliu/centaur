
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizeAndDequantizeV3_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_min_tensor = np.array([-1.0], dtype=np.float32)
    input_max_tensor = np.array([1.0], dtype=np.float32)
    num_bits_tensor = np.array(8, dtype=np.int32)
    signed_input_val = True
    range_given_val = True
    narrow_range_val = False
    axis_val = -1
    name_val = "quantize_dequantize_1"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "num_bits": num_bits_tensor,
        "signed_input": signed_input_val,
        "range_given": range_given_val,
        "narrow_range": narrow_range_val,
        "axis": axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_min_tensor = np.array([0.0], dtype=np.float32)
    input_max_tensor = np.array([4.0], dtype=np.float32)
    num_bits_tensor = np.array(4, dtype=np.int32)
    signed_input_val = False
    range_given_val = True
    narrow_range_val = False
    axis_val = -1
    name_val = "quantize_dequantize_2"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "num_bits": num_bits_tensor,
        "signed_input": signed_input_val,
        "range_given": range_given_val,
        "narrow_range": narrow_range_val,
        "axis": axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    input_min_tensor = np.array([-1.0], dtype=np.float64)
    input_max_tensor = np.array([1.0], dtype=np.float64)
    num_bits_tensor = np.array(2, dtype=np.int32)
    signed_input_val = True
    range_given_val = True
    narrow_range_val = True
    axis_val = -1
    name_val = "quantize_dequantize_3"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "num_bits": num_bits_tensor,
        "signed_input": signed_input_val,
        "range_given": range_given_val,
        "narrow_range": narrow_range_val,
        "axis": axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_min_tensor = np.array([1.0], dtype=np.float32)
    input_max_tensor = np.array([8.0], dtype=np.float32)
    num_bits_tensor = np.array(6, dtype=np.int32)
    signed_input_val = False
    range_given_val = True
    narrow_range_val = False
    axis_val = -1
    name_val = "quantize_dequantize_4"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "num_bits": num_bits_tensor,
        "signed_input": signed_input_val,
        "range_given": range_given_val,
        "narrow_range": narrow_range_val,
        "axis": axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_min_tensor = np.array([0.0], dtype=np.float32)
    input_max_tensor = np.array([5.0], dtype=np.float32)
    num_bits_tensor = np.array(3, dtype=np.int32)
    signed_input_val = False
    range_given_val = False
    narrow_range_val = False
    axis_val = -1
    name_val = "quantize_dequantize_5"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "num_bits": num_bits_tensor,
        "signed_input": signed_input_val,
        "range_given": range_given_val,
        "narrow_range": narrow_range_val,
        "axis": axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    input_min_tensor = np.array([-5.0], dtype=np.float32)
    input_max_tensor = np.array([5.0], dtype=np.float32)
    num_bits_tensor = np.array(7, dtype=np.int32)
    signed_input_val = True
    range_given_val = True
    narrow_range_val = True
    axis_val = -1
    name_val = "quantize_dequantize_6"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "num_bits": num_bits_tensor,
        "signed_input": signed_input_val,
        "range_given": range_given_val,
        "narrow_range": narrow_range_val,
        "axis": axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 Removed the bfloat16 type, as it is not available directly in numpy

    # Input 8
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_min_tensor = np.array([0.0], dtype=np.float32)
    input_max_tensor = np.array([3.0], dtype=np.float32)
    num_bits_tensor = np.array(16, dtype=np.int32)
    signed_input_val = False
    range_given_val = True
    narrow_range_val = False
    axis_val = -1
    name_val = "quantize_dequantize_8"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "num_bits": num_bits_tensor,
        "signed_input": signed_input_val,
        "range_given": range_given_val,
        "narrow_range": narrow_range_val,
        "axis": axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_min_tensor = np.array([0.0], dtype=np.float32)
    input_max_tensor = np.array([3.0], dtype=np.float32)
    num_bits_tensor = np.array(1, dtype=np.int32)
    signed_input_val = False
    range_given_val = True
    narrow_range_val = False
    axis_val = -1
    name_val = "quantize_dequantize_9"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "num_bits": num_bits_tensor,
        "signed_input": signed_input_val,
        "range_given": range_given_val,
        "narrow_range": narrow_range_val,
        "axis": axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_min_tensor = np.array([0.0], dtype=np.float32)
    input_max_tensor = np.array([3.0], dtype=np.float32)
    num_bits_tensor = np.array(8, dtype=np.int32)
    signed_input_val = False
    range_given_val = True
    narrow_range_val = True
    axis_val = -1
    name_val = "quantize_dequantize_10"

    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "num_bits": num_bits_tensor,
        "signed_input": signed_input_val,
        "range_given": range_given_val,
        "narrow_range": narrow_range_val,
        "axis": axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Removed float16 as it does not seem to work consistently
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizeAndDequantizeV3"] = tf_raw_ops_QuantizeAndDequantizeV3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizeAndDequantizeV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeAndDequantizeV3'.")

check_valid('tf.raw_ops.QuantizeAndDequantizeV3', generated_inputs['tf.raw_ops.QuantizeAndDequantizeV3'], lib="tf", suffix=0)
