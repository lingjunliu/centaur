
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_histogram_summary_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.HistogramSummary function.
    """
    list_of_inputs = []

    # Input 1: Basic float32 1D tensor
    input_dict_1 = {
        'tag': np.array("my_float32_hist", dtype=object),
        'values': np.array([1.0, 2.5, -3.0, 4.5, 1.0], dtype=np.float32),
        'name': 'Float32Histogram'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: int32 2D tensor
    input_dict_2 = {
        'tag': np.array("my_int32_hist", dtype=object),
        'values': np.array([[-1, 0, 1], [10, -10, 5]], dtype=np.int32),
        'name': 'Int32Histogram'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: uint8 3D tensor, no optional name
    input_dict_3 = {
        'tag': np.array("uint8_3d_hist", dtype=object),
        'values': np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: float64 scalar value
    input_dict_4 = {
        'tag': np.array("float64_scalar_hist", dtype=object),
        'values': np.array(123.456, dtype=np.float64),
        'name': 'ScalarHistogram'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: int64 1D tensor with large values
    input_dict_5 = {
        'tag': np.array("int64_large_values", dtype=object),
        'values': np.array([10**10, -10**12, 5 * 10**9], dtype=np.int64),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: int8 tensor
    input_dict_6 = {
        'tag': np.array("int8_range", dtype=object),
        'values': np.array([-128, -1, 0, 1, 127], dtype=np.int8),
        'name': 'Int8Hist'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty values tensor
    input_dict_7 = {
        'tag': np.array("empty_hist", dtype=object),
        'values': np.array([], dtype=np.float32),
        'name': 'EmptyHistogram'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: int16 tensor
    input_dict_8 = {
        'tag': np.array("int16_hist", dtype=object),
        'values': np.array([-32768, 0, 32767], dtype=np.int16),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: float32 4D tensor
    input_dict_9 = {
        'tag': np.array("float32_4d_hist", dtype=object),
        'values': np.random.rand(1, 2, 3, 4).astype(np.float32),
        'name': 'Float32_4D'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: int32 tensor with all zeros
    input_dict_10 = {
        'tag': np.array("zeros_hist", dtype=object),
        'values': np.zeros((5, 5), dtype=np.int32),
        'name': 'ZerosHistogram'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: float64 2D tensor
    input_dict_11 = {
        'tag': np.array("float64_2d_hist", dtype=object),
        'values': np.array([[1.1, -2.2], [3.3, -4.4]], dtype=np.float64),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: int64 tensor with repeated values
    input_dict_12 = {
        'tag': np.array("repeated_values_hist", dtype=object),
        'values': np.array([5, 5, 5, 5, 5, 5], dtype=np.int64),
        'name': 'RepeatedValuesHist'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.HistogramSummary"] = get_tf_raw_ops_histogram_summary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.HistogramSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.HistogramSummary'.")

check_valid('tf.raw_ops.HistogramSummary', generated_inputs['tf.raw_ops.HistogramSummary'], lib="tf", suffix=0)
