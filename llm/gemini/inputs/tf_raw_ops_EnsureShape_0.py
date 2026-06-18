
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_EnsureShape_inputs():
    list_of_inputs = []
    
    # Input 1: Scalar int32
    list_of_inputs.append({
        'name': 'ensure_shape_scalar',
        'input': np.array(42, dtype=np.int32),
        'shape': []
    })
    
    # Input 2: 1D float32
    list_of_inputs.append({
        'name': 'ensure_shape_1d',
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'shape': [3]
    })
    
    # Input 3: 2D int64
    list_of_inputs.append({
        'name': 'ensure_shape_2d',
        'input': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
        'shape': [3, 2]
    })
    
    # Input 4: 3D float64
    list_of_inputs.append({
        'name': 'ensure_shape_3d',
        'input': np.zeros((2, 2, 2), dtype=np.float64),
        'shape': [2, 2, 2]
    })
    
    # Input 5: 4D int32
    list_of_inputs.append({
        'name': 'ensure_shape_4d',
        'input': np.ones((1, 3, 4, 5), dtype=np.int32),
        'shape': [1, 3, 4, 5]
    })
    
    # Input 6: 1D bool
    list_of_inputs.append({
        'name': 'ensure_shape_bool',
        'input': np.array([True, False, True], dtype=np.bool_),
        'shape': [3]
    })
    
    # Input 7: 2D float32 with negative values
    list_of_inputs.append({
        'name': 'ensure_shape_neg',
        'input': np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32),
        'shape': [2, 2]
    })
    
    # Input 8: Empty 1D tensor
    list_of_inputs.append({
        'name': 'ensure_shape_empty_1d',
        'input': np.array([], dtype=np.float32),
        'shape': [0]
    })

    # Input 9: Empty 2D tensor
    list_of_inputs.append({
        'name': 'ensure_shape_empty_2d',
        'input': np.empty((2, 0), dtype=np.int32),
        'shape': [2, 0]
    })

    # Input 10: 5D float32
    list_of_inputs.append({
        'name': 'ensure_shape_5d',
        'input': np.ones((1, 2, 1, 2, 1), dtype=np.float32),
        'shape': [1, 2, 1, 2, 1]
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_EnsureShape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.EnsureShape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EnsureShape'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.EnsureShape', generated_inputs['tf.raw_ops.EnsureShape'], lib="tf", suffix=0)
