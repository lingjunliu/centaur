
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizeAndDequantizeV2_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 input
    input_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_min_val = np.array([-3.0], dtype=np.float32)
    input_max_val = np.array([3.0], dtype=np.float32)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": True,
        "num_bits": 8,
        "range_given": True,
        "round_mode": "HALF_TO_EVEN",
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different num_bits and unsigned
    input_val = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    input_min_val = np.array([0.0], dtype=np.float32)
    input_max_val = np.array([1.0], dtype=np.float32)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": False,
        "num_bits": 4,
        "range_given": True,
        "round_mode": "HALF_TO_EVEN",
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Half precision
    input_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    input_min_val = np.array([-3.0], dtype=np.float16)
    input_max_val = np.array([3.0], dtype=np.float16)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": True,
        "num_bits": 8,
        "range_given": True,
        "round_mode": "HALF_TO_EVEN",
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different round_mode
    input_val = np.array([0.4, 0.5, 0.6, 1.4, 1.5, 1.6], dtype=np.float32)
    input_min_val = np.array([0.0], dtype=np.float32)
    input_max_val = np.array([2.0], dtype=np.float32)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": False,
        "num_bits": 8,
        "range_given": True,
        "round_mode": "HALF_UP",
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Narrow range
    input_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_min_val = np.array([-3.0], dtype=np.float32)
    input_max_val = np.array([3.0], dtype=np.float32)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": True,
        "num_bits": 8,
        "range_given": True,
        "round_mode": "HALF_TO_EVEN",
        "narrow_range": True,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: float64 type
    input_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    input_min_val = np.array([-3.0], dtype=np.float64)
    input_max_val = np.array([3.0], dtype=np.float64)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": True,
        "num_bits": 8,
        "range_given": True,
        "round_mode": "HALF_TO_EVEN",
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D Input
    input_val = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    input_min_val = np.array([-3.0], dtype=np.float32)
    input_max_val = np.array([3.0], dtype=np.float32)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": True,
        "num_bits": 8,
        "range_given": True,
        "round_mode": "HALF_TO_EVEN",
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: input_min and input_max are same
    input_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_min_val = np.array([0.0], dtype=np.float32)
    input_max_val = np.array([0.0], dtype=np.float32)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": True,
        "num_bits": 8,
        "range_given": True,
        "round_mode": "HALF_TO_EVEN",
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Axis specified, 2D input
    input_val = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    input_min_val = np.array([-3.0], dtype=np.float32)
    input_max_val = np.array([3.0], dtype=np.float32)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": True,
        "num_bits": 8,
        "range_given": True,
        "round_mode": "HALF_TO_EVEN",
        "narrow_range": False,
        "axis": 1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: range_given = False
    input_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_min_val = np.array([-5.0], dtype=np.float32)
    input_max_val = np.array([5.0], dtype=np.float32)

    input_dict = {
        "input": input_val,
        "input_min": input_min_val,
        "input_max": input_max_val,
        "signed_input": True,
        "num_bits": 8,
        "range_given": False,
        "round_mode": "HALF_TO_EVEN",
        "narrow_range": False,
        "axis": -1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizeAndDequantizeV2"] = tf_raw_ops_QuantizeAndDequantizeV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizeAndDequantizeV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeAndDequantizeV2'.")

check_valid('tf.raw_ops.QuantizeAndDequantizeV2', generated_inputs['tf.raw_ops.QuantizeAndDequantizeV2'], lib="tf", suffix=0)
