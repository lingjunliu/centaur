
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_igammac_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.Igammac function.
    """
    list_of_inputs = []

    # Input 1: Basic float32 scalars
    input_dict_1 = {
        'a': np.array(1.0, dtype=np.float32),
        'x': np.array(2.0, dtype=np.float32),
        'name': 'float32_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic float64 vectors
    input_dict_2 = {
        'a': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'x': np.array([0.5, 1.5, 2.5], dtype=np.float64),
        'name': 'float64_vectors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensors with float32 type
    input_dict_3 = {
        'a': np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        'x': np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32),
        'name': 'float32_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Case where x contains zero
    input_dict_4 = {
        'a': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'x': np.array([0.0, 5.0, 0.0], dtype=np.float32),
        'name': 'x_contains_zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Large values for a and x
    input_dict_5 = {
        'a': np.array([100.0, 150.0], dtype=np.float64),
        'x': np.array([120.0, 130.0], dtype=np.float64),
        'name': 'large_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Small positive values for a and x
    input_dict_6 = {
        'a': np.array([1e-5, 1e-6], dtype=np.float32),
        'x': np.array([1e-4, 1e-5], dtype=np.float32),
        'name': 'small_positive_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 2D matrices with mixed value ranges
    input_dict_7 = {
        'a': np.array([[1.0, 100.0], [0.1, 50.0]], dtype=np.float64),
        'x': np.array([[0.5, 80.0], [0.2, 60.0]], dtype=np.float64),
        'name': None  # Optional name parameter not provided
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single element tensors (vectors)
    input_dict_8 = {
        'a': np.array([5.0], dtype=np.float32),
        'x': np.array([2.0], dtype=np.float32),
        'name': 'single_element_vectors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty tensors
    input_dict_9 = {
        'a': np.array([], dtype=np.float32),
        'x': np.array([], dtype=np.float32),
        'name': 'empty_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Case where a is very close to x
    input_dict_10 = {
        'a': np.array([10.0, 20.0, 30.0], dtype=np.float64),
        'x': np.array([9.99, 19.99, 29.99], dtype=np.float64),
        'name': 'a_close_to_x'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Case where x is much larger than a
    input_dict_11 = {
        'a': np.array([2.0, 3.0], dtype=np.float32),
        'x': np.array([200.0, 300.0], dtype=np.float32),
        'name': 'x_larger_than_a'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: high dimensional input
    input_dict_12 = {
        'a': np.random.rand(2,3,4,5).astype(np.float32),
        'x': np.random.rand(2,3,4,5).astype(np.float32),
        'name': 'high_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))


    return list_of_inputs

generated_inputs["tf.raw_ops.Igammac"] = tf_raw_ops_igammac_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Igammac' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Igammac'.")

check_valid('tf.raw_ops.Igammac', generated_inputs['tf.raw_ops.Igammac'], lib="tf", suffix=0)
