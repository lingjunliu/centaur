
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_identity_n_inputs():
    list_of_inputs = []

    # Input 1: A list of two 1D integer tensors, stacked into a 2D tensor.
    input_dict = {
        'input': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        'name': 'stacked_1d_int_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: A list of two 2D float tensors, stacked into a 3D tensor.
    input_dict = {
        'input': np.array([[[1.0, 2.5], [3.1, 4.2]], [[-1.0, -2.0], [-3.0, -4.0]]], dtype=np.float32),
        'name': 'stacked_2d_float_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A list containing a single 3D tensor.
    input_dict = {
        'input': np.arange(8, dtype=np.float64).reshape((1, 2, 2, 2)),
        'name': 'single_3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: A list of scalar (0D) tensors, stacked into a 1D tensor.
    input_dict = {
        'input': np.array([10, -5, 0], dtype=np.int64),
        'name': 'stacked_scalar_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: A list of high-dimensional (4D) tensors.
    input_dict = {
        'input': np.stack([np.ones((1, 2, 2, 3), dtype=np.int16), np.zeros((1, 2, 2, 3), dtype=np.int16)]),
        'name': 'stacked_high_dim_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A list of 1D complex number tensors.
    input_dict = {
        'input': np.array([[1+2j, 3+4j], [5-6j, 7-8j]], dtype=np.complex64),
        'name': 'stacked_complex_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: A list of 1D boolean tensors.
    input_dict = {
        'input': np.array([[True, False, True], [False, True, False]], dtype=bool),
        'name': 'stacked_boolean_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: A list of multiple 1D tensors (shape (1,)).
    input_dict = {
        'input': np.array([[1], [2], [3], [4], [5]], dtype=np.int8),
        'name': 'many_tensors_in_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A list of float16 tensors.
    input_dict = {
        'input': np.array([[1.0], [2.0], [-3.0]], dtype=np.float16),
        'name': 'stacked_float16_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: A single tensor in the list
    input_dict = {
        'input': np.ones(shape=(1, 5, 5), dtype=np.uint8),
        'name': 'single_tensor_in_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.identity_n"] = tf_identity_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.identity_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity_n'.")

check_valid('tf.identity_n', generated_inputs['tf.identity_n'], lib="tf", suffix=0)
