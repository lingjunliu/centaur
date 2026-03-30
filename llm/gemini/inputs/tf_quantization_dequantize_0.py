
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_quantization_dequantize_inputs():
    list_of_inputs = []

    def create_input(input_tensor, min_range, max_range, mode, name, axis, narrow_range, dtype):
        return {"input": input_tensor, "min_range": min_range, "max_range": max_range, "mode": mode, "name": name, "axis": axis, "narrow_range": narrow_range, "dtype": dtype}

    # Input 1
    input_tensor = np.array([10, 50, 100], dtype=np.int8)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(127.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "MIN_COMBINED", "dequantize_example_1", -1, False, tf.float32)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([10, 50, 100], dtype=np.uint8)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(255.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "MIN_FIRST", "dequantize_example_2", -1, True, tf.float32)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([-10, -50, -100], dtype=np.int8)
    min_range_tensor = np.float32(-128.0)
    max_range_tensor = np.float32(0.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "SCALED", "dequantize_example_3", -1, False, tf.float32)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1000, 5000, 10000], dtype=np.int32)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(2147483647.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "MIN_COMBINED", "dequantize_example_4", -1, False, tf.float32)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[10, 20], [30, 40]], dtype=np.int16)
    min_range_tensor = np.float32(-32768.0)
    max_range_tensor = np.float32(32767.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "MIN_FIRST", "dequantize_example_5", -1, False, tf.float32)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(65535.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "SCALED", "dequantize_example_6", -1, False, tf.float32)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([10, 50, 100], dtype=np.int8)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(127.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "MIN_COMBINED", "dequantize_example_7", 0, False, tf.float32)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([10, 50, 100], dtype=np.uint8)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(255.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "MIN_FIRST", "dequantize_example_8", 0, True, tf.bfloat16)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([-10, -50, -100], dtype=np.int8)
    min_range_tensor = np.float32(-128.0)
    max_range_tensor = np.float32(0.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "MIN_COMBINED", "dequantize_example_9", 0, False, tf.float32)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int16)
    min_range_tensor = np.float32(-1.0)
    max_range_tensor = np.float32(1.0)
    input_dict = create_input(input_tensor, min_range_tensor, max_range_tensor, "MIN_COMBINED", "dequantize_example_10", -1, False, tf.float32)
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.quantization.dequantize"] = tf_quantization_dequantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.quantization.dequantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.dequantize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.quantization.dequantize', generated_inputs['tf.quantization.dequantize'], lib="tf", suffix=0)
