
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_range_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integer range (int32)
    input_dict_1 = {
        'name': 'basic_int32_range',
        'start': np.array(0, dtype=np.int32),
        'limit': np.array(10, dtype=np.int32),
        'delta': np.array(1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Integer range with a step > 1 (from example, int32)
    input_dict_2 = {
        'name': 'int32_range_step_3',
        'start': np.array(3, dtype=np.int32),
        'limit': np.array(18, dtype=np.int32),
        'delta': np.array(3, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Basic float32 range
    input_dict_3 = {
        'name': 'float32_range',
        'start': np.array(0.0, dtype=np.float32),
        'limit': np.array(1.0, dtype=np.float32),
        'delta': np.array(0.1, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Decreasing integer range (negative delta, int32)
    input_dict_4 = {
        'name': 'decreasing_int32_range',
        'start': np.array(10, dtype=np.int32),
        'limit': np.array(0, dtype=np.int32),
        'delta': np.array(-2, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Empty range (start == limit, int32)
    input_dict_5 = {
        'name': 'empty_range_equal',
        'start': np.array(5, dtype=np.int32),
        'limit': np.array(5, dtype=np.int32),
        'delta': np.array(1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Float64 range
    input_dict_6 = {
        'name': 'float64_range',
        'start': np.array(0.5, dtype=np.float64),
        'limit': np.array(5.0, dtype=np.float64),
        'delta': np.array(0.5, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Large integer range using int64
    input_dict_7 = {
        'name': 'large_int64_range',
        'start': np.array(1000000000, dtype=np.int64),
        'limit': np.array(1000000010, dtype=np.int64),
        'delta': np.array(2, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Decreasing float range (float32)
    input_dict_8 = {
        'name': 'decreasing_float32_range',
        'start': np.array(5.0, dtype=np.float32),
        'limit': np.array(-5.0, dtype=np.float32),
        'delta': np.array(-1.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: float16 (half) range
    input_dict_9 = {
        'name': 'float16_range',
        'start': np.array(0.0, dtype=np.float16),
        'limit': np.array(10.0, dtype=np.float16),
        'delta': np.array(1.5, dtype=np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Negative int64 range
    input_dict_10 = {
        'name': 'negative_int64_range',
        'start': np.array(-20, dtype=np.int64),
        'limit': np.array(0, dtype=np.int64),
        'delta': np.array(3, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Decreasing float64 range
    input_dict_11 = {
        'name': 'decreasing_float64_range',
        'start': np.array(10.0, dtype=np.float64),
        'limit': np.array(-10.0, dtype=np.float64),
        'delta': np.array(-2.5, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.Range"] = tf_raw_ops_range_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Range' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Range'.")

check_valid('tf.raw_ops.Range', generated_inputs['tf.raw_ops.Range'], lib="tf", suffix=0)
