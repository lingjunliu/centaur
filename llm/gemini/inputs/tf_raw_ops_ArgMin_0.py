
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ArgMin_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        "input": np.array([1.0, 10.0, 26.9, 2.8, 166.32, 62.3], dtype=np.float32),
        "dimension": np.array(0, dtype=np.int32),
        "output_type": tf.int64,
        "name": "argmin_1"
    })
    
    # Input 2
    list_of_inputs.append({
        "input": np.array([[1, 2, 3], [4, 5, -1]], dtype=np.int32),
        "dimension": np.array(1, dtype=np.int32),
        "output_type": tf.int32,
        "name": "argmin_2"
    })

    # Input 3
    list_of_inputs.append({
        "input": np.array([[[10], [20]], [[30], [5]], [[40], [50]]], dtype=np.uint8),
        "dimension": np.array(0, dtype=np.int64),
        "output_type": tf.int64,
        "name": "argmin_3"
    })

    # Input 4
    list_of_inputs.append({
        "input": np.array([-1.5, -2.5, -0.5, -10.2], dtype=np.float64),
        "dimension": np.array(0, dtype=np.int32),
        "output_type": tf.int32,
        "name": "argmin_4"
    })

    # Input 5
    list_of_inputs.append({
        "input": np.array([[True, False], [False, False]], dtype=np.bool_),
        "dimension": np.array(0, dtype=np.int32),
        "output_type": tf.int64,
        "name": "argmin_5"
    })

    # Input 6
    list_of_inputs.append({
        "input": np.array([[10, 20, 30], [5, 2, 1]], dtype=np.int16),
        "dimension": np.array(1, dtype=np.int64),
        "output_type": tf.int32,
        "name": "argmin_6"
    })

    # Input 7
    list_of_inputs.append({
        "input": np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int64),
        "dimension": np.array(2, dtype=np.int32),
        "output_type": tf.int64,
        "name": "argmin_7"
    })

    # Input 8
    list_of_inputs.append({
        "input": np.random.uniform(-10.0, 10.0, size=(5,)).astype(np.float32),
        "dimension": np.array(-1, dtype=np.int32),
        "output_type": tf.int32,
        "name": "argmin_8"
    })

    # Input 9
    list_of_inputs.append({
        "input": np.array([[3, 3, 3], [3, 3, 3]], dtype=np.int32),
        "dimension": np.array(1, dtype=np.int32),
        "output_type": tf.int64,
        "name": "argmin_9"
    })

    # Input 10
    list_of_inputs.append({
        "input": np.ones((2, 2, 2, 2), dtype=np.float32),
        "dimension": np.array(3, dtype=np.int32),
        "output_type": tf.int32,
        "name": "argmin_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ArgMin"] = tf_raw_ops_ArgMin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ArgMin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ArgMin'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ArgMin', generated_inputs['tf.raw_ops.ArgMin'], lib="tf", suffix=0)
