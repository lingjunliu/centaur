
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_random_stateless_binomial_inputs():
    """
    Generates a list of valid inputs for tf.random.stateless_binomial.
    """
    list_of_inputs = []

    # Input 1: Basic case with scalar counts and probs
    input_dict_1 = {
        'shape': np.array([2, 3], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int32),
        'counts': np.array(10.0, dtype=np.float32),
        'probs': np.array(0.5, dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'basic_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Vector counts and probs where shape equals broadcasted shape
    input_dict_2 = {
        'shape': np.array([4], dtype=np.int32),
        'seed': np.array([123, 456], dtype=np.int64),
        'counts': np.array([10., 20., 30., 40.], dtype=np.float32),
        'probs': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        'output_dtype': np.int64,
        'name': 'vector_params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Broadcasting of counts and probs
    input_dict_3 = {
        'shape': np.array([2, 3], dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int32),
        'counts': np.array([10., 20., 30.], dtype=np.float32),  # Shape [3]
        'probs': np.array([[0.8], [0.9]], dtype=np.float32),  # Shape [2, 1]
        'output_dtype': np.int32,
        'name': 'broadcast_params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Higher dimensional shape with broadcasting
    input_dict_4 = {
        'shape': np.array([2, 2, 3], dtype=np.int32),
        'seed': np.array([99, 100], dtype=np.int32),
        'counts': np.array([5., 15., 25.], dtype=np.float32),
        'probs': np.array(0.7, dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'high_dim_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: First example from documentation
    input_dict_5 = {
        'shape': np.array([2], dtype=np.int32),
        'seed': np.array([123, 456], dtype=np.int32),
        'counts': np.array([10., 20.], dtype=np.float32),
        'probs': np.array([0.8], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'doc_example_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Second example from documentation (complex broadcasting)
    input_dict_6 = {
        'shape': np.array([3, 4, 3, 4, 2], dtype=np.int32),
        'seed': np.array([123, 456], dtype=np.int32),
        'counts': np.array([[[10., 20.]], [[30., 40.]], [[50., 60.]]], dtype=np.float32),  # Shape [3, 1, 2]
        'probs': np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]], dtype=np.float32), # Shape [1, 4, 2]
        'output_dtype': np.int32,
        'name': 'doc_example_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Using float64 for counts and probs
    input_dict_7 = {
        'shape': np.array([5, 1], dtype=np.int32),
        'seed': np.array([2**33, 2**34], dtype=np.int64),
        'counts': np.array([100.], dtype=np.float64),
        'probs': np.array([0.99], dtype=np.float64),
        'output_dtype': np.int32,
        'name': 'float64_params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Zero counts
    input_dict_8 = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32),
        'counts': np.array([[0., 10.], [0., 20.]], dtype=np.float32),
        'probs': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'zero_counts'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Edge case probabilities (0 and 1)
    input_dict_9 = {
        'shape': np.array([4], dtype=np.int32),
        'seed': np.array([42, 42], dtype=np.int32),
        'counts': np.array([10., 10., 10., 10.], dtype=np.float32),
        'probs': np.array([0., 1., 0., 1.], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'edge_probs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Matching float32 dtypes for counts and probs
    input_dict_10 = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([101, 102], dtype=np.int32),
        'counts': np.array([[10, 20], [30, 40]], dtype=np.float32),
        'probs': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'matching_float_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Broadcasting to match rightmost dimensions of shape
    input_dict_11 = {
        'shape': np.array([5, 2, 3], dtype=np.int32),
        'seed': np.array([11, 22], dtype=np.int32),
        'counts': np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32), # Shape [2, 3]
        'probs': np.array([0.1, 0.2, 0.3], dtype=np.float32), # Shape [3]
        'output_dtype': np.int32,
        'name': 'broadcast_to_rightmost_dims'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Empty shape for scalar output
    input_dict_12 = {
        'shape': np.array([], dtype=np.int32),
        'seed': np.array([777, 888], dtype=np.int32),
        'counts': np.array(10., dtype=np.float32),
        'probs': np.array(0.25, dtype=np.float32),
        'output_dtype': np.int64,
        'name': 'empty_shape_scalar_output'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.random.stateless_binomial"] = get_tf_random_stateless_binomial_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_binomial' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_binomial'.")

check_valid('tf.random.stateless_binomial', generated_inputs['tf.random.stateless_binomial'], lib="tf", suffix=0)
