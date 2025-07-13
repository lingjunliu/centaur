
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizeAndDequantizeV3_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 example
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "input_min": np.array([-2.0], dtype=np.float32),
        "input_max": np.array([2.0], dtype=np.float32),
        "num_bits": np.array(8, dtype=np.int32),
        "signed_input": True,
        "range_given": True,
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with different num_bits
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        "input_min": np.array([-2.0], dtype=np.float64),
        "input_max": np.array([2.0], dtype=np.float64),
        "num_bits": np.array(4, dtype=np.int32),
        "signed_input": True,
        "range_given": True,
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  float16 example
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0], dtype=np.float16),
        "input_min": np.array([-2.0], dtype=np.float16),
        "input_max": np.array([2.0], dtype=np.float16),
        "num_bits": np.array(8, dtype=np.int32),
        "signed_input": True,
        "range_given": True,
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array
    input_dict = {
        "input": np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32),
        "input_min": np.array([-2.0], dtype=np.float32),
        "input_max": np.array([2.0], dtype=np.float32),
        "num_bits": np.array(8, dtype=np.int32),
        "signed_input": True,
        "range_given": True,
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: unsigned input
    input_dict = {
        "input": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "input_min": np.array([0.0], dtype=np.float32),
        "input_max": np.array([2.0], dtype=np.float32),
        "num_bits": np.array(8, dtype=np.int32),
        "signed_input": False,
        "range_given": True,
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  narrow range
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "input_min": np.array([-2.0], dtype=np.float32),
        "input_max": np.array([2.0], dtype=np.float32),
        "num_bits": np.array(8, dtype=np.int32),
        "signed_input": True,
        "range_given": True,
        "narrow_range": True,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: range not given
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "input_min": np.array([-2.0], dtype=np.float32),
        "input_max": np.array([2.0], dtype=np.float32),
        "num_bits": np.array(8, dtype=np.int32),
        "signed_input": True,
        "range_given": False,
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different axis
    input_dict = {
        "input": np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32),
        "input_min": np.array([-2.0], dtype=np.float32),
        "input_max": np.array([2.0], dtype=np.float32),
        "num_bits": np.array(8, dtype=np.int32),
        "signed_input": True,
        "range_given": True,
        "narrow_range": False,
        "axis": 0,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: large number of bits
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "input_min": np.array([-2.0], dtype=np.float32),
        "input_max": np.array([2.0], dtype=np.float32),
        "num_bits": np.array(16, dtype=np.int32),
        "signed_input": True,
        "range_given": True,
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizeAndDequantizeV3"] = tf_raw_ops_QuantizeAndDequantizeV3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizeAndDequantizeV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeAndDequantizeV3'.")

check_valid('tf.raw_ops.QuantizeAndDequantizeV3', generated_inputs['tf.raw_ops.QuantizeAndDequantizeV3'], lib="tf", suffix=0)
