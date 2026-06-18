
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Bucketize_inputs():
    list_of_inputs = []

    # Input 1: Float32 1D input, simple boundaries
    list_of_inputs.append({
        'name': 'bucketize_1',
        'input': np.array([-1.5, 0.0, 1.5, 2.5, 10.0], dtype=np.float32),
        'boundaries': [0.0, 2.0]
    })

    # Input 2: Float32 2D input, matching the doc example
    list_of_inputs.append({
        'name': 'bucketize_2',
        'input': np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.float32),
        'boundaries': [0.0, 10.0, 100.0]
    })

    # Input 3: Int32 2D input, negative boundaries
    list_of_inputs.append({
        'name': 'bucketize_3',
        'input': np.array([[-10, 0], [10, 20]], dtype=np.int32),
        'boundaries': [-5.0, 5.0, 15.0]
    })

    # Input 4: Float64 3D input
    list_of_inputs.append({
        'name': 'bucketize_4',
        'input': np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float64),
        'boundaries': [2.0, 4.0, 6.0, 8.0]
    })

    # Input 5: Int64 1D input, large values
    list_of_inputs.append({
        'name': 'bucketize_5',
        'input': np.array([100, 200, 300, 400], dtype=np.int64),
        'boundaries': [150.0, 250.0, 350.0]
    })

    # Input 6: Float32 1D input, empty boundaries
    list_of_inputs.append({
        'name': 'bucketize_6',
        'input': np.array([-1.0, 1.0], dtype=np.float32),
        'boundaries': []
    })

    # Input 7: Int32 Scalar (0D) input
    list_of_inputs.append({
        'name': 'bucketize_7',
        'input': np.array(5, dtype=np.int32),
        'boundaries': [0.0, 10.0]
    })

    # Input 8: Float64 2D input with many boundaries
    list_of_inputs.append({
        'name': 'bucketize_8',
        'input': np.array([[0.1, 0.9], [0.4, 0.6]], dtype=np.float64),
        'boundaries': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    })

    # Input 9: Int64 4D input
    list_of_inputs.append({
        'name': 'bucketize_9',
        'input': np.ones((2, 2, 2, 2), dtype=np.int64) * 10,
        'boundaries': [5.0, 15.0]
    })

    # Input 10: Float32 3D input with boundaries containing large ranges
    list_of_inputs.append({
        'name': 'bucketize_10',
        'input': np.array([[[1e-5, 1e5]]], dtype=np.float32),
        'boundaries': [1e-3, 1.0, 1e3]
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Bucketize"] = tf_raw_ops_Bucketize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Bucketize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bucketize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Bucketize', generated_inputs['tf.raw_ops.Bucketize'], lib="tf", suffix=0)
