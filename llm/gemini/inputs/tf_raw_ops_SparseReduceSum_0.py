
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseReduceSum_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test1",
        'input_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'input_values': np.array([1.5, 2.5], dtype=np.float32),
        'input_shape': np.array([2, 3], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    })
    
    # Input 2
    list_of_inputs.append({
        'keep_dims': True,
        'name': "test2",
        'input_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'input_values': np.array([1.5, 2.5], dtype=np.float32),
        'input_shape': np.array([2, 3], dtype=np.int64),
        'reduction_axes': np.array([1], dtype=np.int32)
    })
    
    # Input 3
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test3",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([1.0, 2.0], dtype=np.float32),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([0, 1], dtype=np.int32)
    })
    
    # Input 4
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test4",
        'input_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'input_values': np.array([1.5, 2.5], dtype=np.float32),
        'input_shape': np.array([2, 3], dtype=np.int64),
        'reduction_axes': np.array([-1], dtype=np.int32)
    })
    
    # Input 5
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test5",
        'input_indices': np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        'input_values': np.array([10, 20], dtype=np.int32),
        'input_shape': np.array([2, 2, 2], dtype=np.int64),
        'reduction_axes': np.array([0, 2], dtype=np.int32)
    })
    
    # Input 6
    list_of_inputs.append({
        'keep_dims': True,
        'name': "test6",
        'input_indices': np.array([[0], [2], [4]], dtype=np.int64),
        'input_values': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'input_shape': np.array([5], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    })
    
    # Input 7
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test7",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([1.0, 2.0], dtype=np.float32),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([], dtype=np.int32)
    })
    
    # Input 8
    list_of_inputs.append({
        'keep_dims': True,
        'name': "test8",
        'input_indices': np.array([[0, 1, 2, 3]], dtype=np.int64),
        'input_values': np.array([5.0], dtype=np.float32),
        'input_shape': np.array([2, 3, 4, 5], dtype=np.int64),
        'reduction_axes': np.array([1, 3], dtype=np.int32)
    })
    
    # Input 9
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test9",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([1], dtype=np.int32)
    })
    
    # Input 10
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test10",
        'input_indices': np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),
        'input_values': np.array([100, 200, 300], dtype=np.int64),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    })
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SparseReduceSum"] = tf_raw_ops_SparseReduceSum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseReduceSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseReduceSum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseReduceSum', generated_inputs['tf.raw_ops.SparseReduceSum'], lib="tf", suffix=0)
