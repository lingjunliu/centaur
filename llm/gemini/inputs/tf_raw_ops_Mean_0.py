
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Mean_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_1',
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'axis': np.array(0, dtype=np.int32)
    })
    
    # Input 2
    list_of_inputs.append({
        'keep_dims': True,
        'name': 'mean_2',
        'input': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'axis': np.array([0], dtype=np.int32)
    })
    
    # Input 3
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_3',
        'input': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'axis': np.array([1], dtype=np.int64)
    })
    
    # Input 4
    list_of_inputs.append({
        'keep_dims': True,
        'name': 'mean_4',
        'input': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        'axis': np.array([0, 2], dtype=np.int32)
    })
    
    # Input 5
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_5',
        'input': np.array([10, 20, 30], dtype=np.int64),
        'axis': np.array([-1], dtype=np.int32)
    })
    
    # Input 6
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_6',
        'input': np.array([1.5, 3.5], dtype=np.float32),
        'axis': np.array(0, dtype=np.int32)
    })
    
    # Input 7
    list_of_inputs.append({
        'keep_dims': True,
        'name': 'mean_7',
        'input': np.array([[1, 2, 3]], dtype=np.int64),
        'axis': np.array([1], dtype=np.int32)
    })
    
    # Input 8
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_8',
        'input': np.array([1.0, 2.0], dtype=np.float32),
        'axis': np.array(0, dtype=np.int64)
    })
    
    # Input 9
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_9',
        'input': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'axis': np.array([0, 1], dtype=np.int32)
    })
    
    # Input 10
    list_of_inputs.append({
        'keep_dims': True,
        'name': 'mean_10',
        'input': np.array([10.0, 20.0], dtype=np.float64),
        'axis': np.array([], dtype=np.int32)
    })
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Mean"] = tf_raw_ops_Mean_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Mean' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Mean'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Mean', generated_inputs['tf.raw_ops.Mean'], lib="tf", suffix=0)
