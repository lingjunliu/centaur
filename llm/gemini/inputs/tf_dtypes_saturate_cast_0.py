
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dtypes_saturate_cast_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "value": np.array([1e10, -1e10, 5.5], dtype=np.float32),
        "dtype": np.int32,
        "name": "cast_1"
    })

    # Input 2
    list_of_inputs.append({
        "value": np.array([[-10.5, 256.1], [100.0, -1.0]], dtype=np.float32),
        "dtype": np.uint8,
        "name": "cast_2"
    })

    # Input 3
    list_of_inputs.append({
        "value": np.array(40000.0, dtype=np.float64),
        "dtype": np.int16,
        "name": "cast_3"
    })

    # Input 4
    list_of_inputs.append({
        "value": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64),
        "dtype": np.float32,
        "name": "cast_4"
    })

    # Input 5
    list_of_inputs.append({
        "value": np.array([150, -200, 50, -50], dtype=np.int32),
        "dtype": np.int8,
        "name": "cast_5"
    })

    # Input 6
    list_of_inputs.append({
        "value": np.array([[[[1000.0], [2000.0]], [[3000.0], [4000.0]]]], dtype=np.float32),
        "dtype": np.float16,
        "name": "cast_6"
    })

    # Input 7
    list_of_inputs.append({
        "value": np.array([[70000, -5], [100, 500]], dtype=np.int32),
        "dtype": np.uint16,
        "name": "cast_7"
    })

    # Input 8
    list_of_inputs.append({
        "value": np.array([np.inf, -np.inf, 1.5], dtype=np.float64),
        "dtype": np.float32,
        "name": "cast_8"
    })

    # Input 9
    list_of_inputs.append({
        "value": np.array([[[128, -129], [0, 1]], [[127, -128], [5, -5]]], dtype=np.int32),
        "dtype": np.int8,
        "name": "cast_9"
    })

    # Input 10
    list_of_inputs.append({
        "value": np.array([[5e9, -100], [10, 20]], dtype=np.float32),
        "dtype": np.uint32,
        "name": "cast_10"
    })

    return list_of_inputs

generated_inputs["tf.dtypes.saturate_cast"] = tf_dtypes_saturate_cast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.dtypes.saturate_cast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.dtypes.saturate_cast'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.dtypes.saturate_cast', generated_inputs['tf.dtypes.saturate_cast'], lib="tf", suffix=0)
