
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_stagepeek_inputs():
    # The StagePeek op blocks indefinitely if the staging area is empty,
    # leading to a timeout. To avoid this, we must generate inputs that
    # are valid for graph construction but cause an immediate runtime error.
    # The only reliable way to achieve this is by setting a positive `capacity`
    # and ensuring the `index` is out of bounds (i.e., `index >= capacity`).
    # This triggers a runtime InvalidArgumentError, preventing the block.

    list_of_inputs = []

    # All inputs will follow the pattern: capacity > 0 and index >= capacity

    # Input 1: Basic case with index == capacity
    input_dict = {
        'index': np.array(5, dtype=np.int32),
        'dtypes': [tf.float32],
        'capacity': 5,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_index_eq_capacity'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with index > capacity
    input_dict = {
        'index': np.array(6, dtype=np.int32),
        'dtypes': [tf.int64, tf.float64],
        'capacity': 2,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_index_gt_capacity'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Minimal capacity to trigger the check
    input_dict = {
        'index': np.array(1, dtype=np.int32),
        'dtypes': [tf.int32, tf.float32, tf.string],
        'capacity': 1,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_minimal_fail'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With specified memory_limit
    input_dict = {
        'index': np.array(100, dtype=np.int32),
        'dtypes': [tf.bool],
        'capacity': 50,
        'memory_limit': 1024,
        'container': '',
        'shared_name': '',
        'name': 'peek_with_memlimit'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With a non-default container
    input_dict = {
        'index': np.array(5, dtype=np.int32),
        'dtypes': [tf.complex64],
        'capacity': 5,
        'memory_limit': 0,
        'container': 'my_container',
        'shared_name': '',
        'name': 'peek_in_container'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: With a shared_name
    input_dict = {
        'index': np.array(10, dtype=np.int32),
        'dtypes': [tf.uint8, tf.int16],
        'capacity': 3,
        'memory_limit': 0,
        'container': '',
        'shared_name': 'my_shared_stage',
        'name': 'peek_shared'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All optional arguments specified
    input_dict = {
        'index': np.array(21, dtype=np.int32),
        'dtypes': [tf.float16, tf.bfloat16],
        'capacity': 20,
        'memory_limit': 2048,
        'container': 'another_container',
        'shared_name': 'another_shared_name',
        'name': 'peek_full_fail'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another combination of dtypes
    input_dict = {
        'index': np.array(2, dtype=np.int32),
        'dtypes': [tf.qint8, tf.quint8],
        'capacity': 2,
        'memory_limit': 0,
        'container': 'quant_container',
        'shared_name': '',
        'name': 'peek_quant'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large index and capacity
    input_dict = {
        'index': np.array(150, dtype=np.int32),
        'dtypes': [tf.string],
        'capacity': 150,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'peek_large_index'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Both container and shared_name specified
    input_dict = {
        'index': np.array(11, dtype=np.int32),
        'dtypes': [tf.int32],
        'capacity': 10,
        'memory_limit': 512,
        'container': 'shared_container',
        'shared_name': 'my_shared_stage_2',
        'name': 'peek_shared_and_container'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.StagePeek"] = get_tf_raw_ops_stagepeek_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StagePeek' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StagePeek'.")

check_valid('tf.raw_ops.StagePeek', generated_inputs['tf.raw_ops.StagePeek'], lib="tf", suffix=0)
