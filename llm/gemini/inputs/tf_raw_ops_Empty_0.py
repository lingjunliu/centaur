
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Empty_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'shape': np.array([2, 3], dtype=np.int32),
        'dtype': np.float32,
        'init': True,
        'name': "empty_1"
    })
    
    # Input 2
    list_of_inputs.append({
        'shape': np.array([5], dtype=np.int32),
        'dtype': np.int32,
        'init': False,
        'name': "empty_2"
    })
    
    # Input 3
    list_of_inputs.append({
        'shape': np.array([0, 4], dtype=np.int32),
        'dtype': np.float64,
        'init': True,
        'name': "empty_3"
    })
    
    # Input 4
    list_of_inputs.append({
        'shape': np.array([1, 1, 1], dtype=np.int32),
        'dtype': np.bool_,
        'init': False,
        'name': "empty_4"
    })
    
    # Input 5
    list_of_inputs.append({
        'shape': np.array([2, 2, 2, 2], dtype=np.int32),
        'dtype': np.int64,
        'init': True,
        'name': "empty_5"
    })
    
    # Input 6
    list_of_inputs.append({
        'shape': np.array([3, 2, 4], dtype=np.int32),
        'dtype': np.uint8,
        'init': False,
        'name': "empty_6"
    })
    
    # Input 7
    list_of_inputs.append({
        'shape': np.array([], dtype=np.int32),
        'dtype': np.float32,
        'init': True,
        'name': "empty_7"
    })
    
    # Input 8
    list_of_inputs.append({
        'shape': np.array([2, 0], dtype=np.int32),
        'dtype': np.complex64,
        'init': False,
        'name': "empty_8"
    })
    
    # Input 9
    list_of_inputs.append({
        'shape': np.array([4, 2], dtype=np.int32),
        'dtype': np.int16,
        'init': True,
        'name': "empty_9"
    })
    
    # Input 10
    list_of_inputs.append({
        'shape': np.array([100], dtype=np.int32),
        'dtype': np.float16,
        'init': False,
        'name': "empty_10"
    })
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Empty"] = tf_raw_ops_Empty_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Empty' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Empty'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Empty', generated_inputs['tf.raw_ops.Empty'], lib="tf", suffix=0)
