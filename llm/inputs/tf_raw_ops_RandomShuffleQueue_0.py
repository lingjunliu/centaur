
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_tf_raw_ops_RandomShuffleQueue_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.RandomShuffleQueue operation.
    NOTE: This operation is designed for TensorFlow's graph mode and will raise a
    RuntimeError if executed eagerly, which is the default in TF 2.x. The provided
    inputs are syntactically correct according to the function's signature but are
    expected to fail in an eager execution environment.
    """
    # Helper to get numpy dtypes from tensorflow string representation
    def to_np_dtype(s):
        return tf.as_dtype(s).as_numpy_dtype

    list_of_inputs = []

    # Input 1: Basic case with minimal configuration.
    input_1 = {
        'component_types': [to_np_dtype('float32')],
        'shapes': [[1]],
        'capacity': 10,
        'min_after_dequeue': 2,
        'seed': 0,
        'seed2': 0,
        'container': '',
        'shared_name': '',
        'name': 'queue_minimal'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Multiple component types.
    input_2 = {
        'component_types': [to_np_dtype('int32'), to_np_dtype('string')],
        'shapes': [[2, 2], []],
        'capacity': 100,
        'min_after_dequeue': 10,
        'seed': 1,
        'seed2': 1,
        'container': 'multi_container',
        'shared_name': 'multi_shared',
        'name': 'queue_multi_type'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Unconstrained shapes and no capacity limit.
    input_3 = {
        'component_types': [to_np_dtype('bool')],
        'shapes': [],
        'capacity': -1,
        'min_after_dequeue': 50,
        'seed': 42,
        'seed2': 0,
        'container': '',
        'shared_name': '',
        'name': 'queue_unconstrained'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Scalar shape and different data type.
    input_4 = {
        'component_types': [to_np_dtype('int64')],
        'shapes': [[]],
        'capacity': 50,
        'min_after_dequeue': 0,
        'seed': 123,
        'seed2': 456,
        'container': '',
        'shared_name': 'scalar_queue',
        'name': 'queue_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_4))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.RandomShuffleQueue"] = generate_tf_raw_ops_RandomShuffleQueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RandomShuffleQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomShuffleQueue'.")

check_valid('tf.raw_ops.RandomShuffleQueue', generated_inputs['tf.raw_ops.RandomShuffleQueue'], lib="tf", suffix=0)
