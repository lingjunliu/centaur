
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scalar_summary_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 values
    input_dict = {
        'tags': np.array(["loss", "accuracy"], dtype=object),
        'values': np.array([0.123, 0.987], dtype=np.float32),
        'name': "training_summary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar (0-D) int32 value
    input_dict = {
        'tags': np.array("epoch_number", dtype=object),
        'values': np.array(10, dtype=np.int32),
        'name': "epoch_summary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 values
    input_dict = {
        'tags': np.array([["layer1_norm", "layer2_norm"], ["layer1_grad", "layer2_grad"]], dtype=object),
        'values': np.array([[1.5, 2.3], [0.05, 0.02]], dtype=np.float64),
        'name': "layer_details"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int64 values with negative numbers
    input_dict = {
        'tags': np.array(["value_a", "value_b"], dtype=object),
        'values': np.array([-10000000000, 90000000000], dtype=np.int64),
        'name': "large_integer_summary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D uint8 values
    input_dict = {
        'tags': np.array(["pixel_min", "pixel_max", "pixel_mean"], dtype=object),
        'values': np.array([0, 255, 128], dtype=np.uint8),
        'name': "image_stats_summary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int16 values with negative numbers
    input_dict = {
        'tags': np.array([["q1", "q2"], ["q3", "q4"]], dtype=object),
        'values': np.array([[-32768, 32767], [0, -1]], dtype=np.int16),
        'name': "quantization_summary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float16 (half) values
    input_dict = {
        'tags': np.array([[['a'], ['b']], [['c'], ['d']]], dtype=object),
        'values': np.array([[[1.1], [2.2]], [[-3.3], [-4.4]]], dtype=np.float16),
        'name': "3d_summary_half"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tensors with matching shapes
    input_dict = {
        'tags': np.array([], dtype=object),
        'values': np.array([], dtype=np.float32),
        'name': "empty_summary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int64 values (changed from uint32 to avoid validation error)
    input_dict = {
        'tags': np.array(["global_step", "total_steps"], dtype=object),
        'values': np.array([50000, 1000000], dtype=np.int64),
        'name': "step_counters"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar int8 value
    input_dict = {
        'tags': np.array("temperature_celcius", dtype=object),
        'values': np.array(-15, dtype=np.int8),
        'name': "weather_summary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 1D int64 values (changed from uint64 to avoid validation error)
    input_dict = {
        'tags': np.array(["disk_free_bytes", "disk_total_bytes"], dtype=object),
        'values': np.array([1234567890123, 9876543210987], dtype=np.int64),
        'name': "system_summary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty 2D tensors with matching shapes
    input_dict = {
        'tags': np.empty((0, 2), dtype=object),
        'values': np.empty((0, 2), dtype=np.int32),
        'name': "empty_2d_summary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["tf.raw_ops.ScalarSummary"] = tf_raw_ops_scalar_summary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScalarSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScalarSummary'.")

check_valid('tf.raw_ops.ScalarSummary', generated_inputs['tf.raw_ops.ScalarSummary'], lib="tf", suffix=0)
