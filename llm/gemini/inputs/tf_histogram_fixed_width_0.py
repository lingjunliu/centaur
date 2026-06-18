
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_histogram_fixed_width_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'values': np.array([-1.0, 0.0, 1.5, 2.0, 5.0, 15], dtype=np.float32),
        'value_range': np.array([0.0, 5.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'hist_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'values': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'value_range': np.array([0.0, 10.0], dtype=np.float64),
        'nbins': 10,
        'dtype': np.int64,
        'name': 'hist_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'values': np.array([-10, -5, 0, 5, 10], dtype=np.int32),
        'value_range': np.array([-15, 15], dtype=np.int32),
        'nbins': 3,
        'dtype': np.int32,
        'name': 'hist_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'values': np.random.uniform(-1.0, 1.0, size=(10, 10)).astype(np.float32),
        'value_range': np.array([-1.0, 1.0], dtype=np.float32),
        'nbins': 20,
        'dtype': np.int32,
        'name': 'hist_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'values': np.random.normal(0, 1, size=(5, 5, 5)).astype(np.float32),
        'value_range': np.array([-3.0, 3.0], dtype=np.float32),
        'nbins': 15,
        'dtype': np.int64,
        'name': 'hist_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'values': np.array([100.5], dtype=np.float32),
        'value_range': np.array([100.0, 101.0], dtype=np.float32),
        'nbins': 2,
        'dtype': np.int32,
        'name': 'hist_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'values': np.arange(10, dtype=np.float64),
        'value_range': np.array([0.0, 9.0], dtype=np.float64),
        'nbins': 9,
        'dtype': np.int32,
        'name': 'hist_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'values': np.array([[[[1, 2], [3, 4]]]], dtype=np.int32),
        'value_range': np.array([1, 4], dtype=np.int32),
        'nbins': 4,
        'dtype': np.int64,
        'name': 'hist_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'values': np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6], dtype=np.float32),
        'value_range': np.array([1.0, 2.0], dtype=np.float32),
        'nbins': 10,
        'dtype': np.int32,
        'name': 'hist_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'values': np.linspace(-100, 100, 50, dtype=np.float32),
        'value_range': np.array([-50.0, 50.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'hist_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.histogram_fixed_width"] = tf_histogram_fixed_width_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.histogram_fixed_width' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.histogram_fixed_width'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.histogram_fixed_width', generated_inputs['tf.histogram_fixed_width'], lib="tf", suffix=0)
