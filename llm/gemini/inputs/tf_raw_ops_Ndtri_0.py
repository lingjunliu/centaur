
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_ndtri_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    input_dict = {
        'x': np.array([0.1, 0.5, 0.9], dtype=np.float32),
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor with 2 dimensions
    input_dict = {
        'x': np.array([[0.25, 0.75], [0.01, 0.99]], dtype=np.float64),
        'name': 'test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0-D (scalar) float32 tensor
    input_dict = {
        'x': np.array(0.5, dtype=np.float32),
        'name': 'scalar_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3-D float32 tensor with random values
    # Using uniform distribution to ensure values are in (0, 1)
    input_dict = {
        'x': np.random.uniform(low=1e-6, high=1.0 - 1e-6, size=(2, 3, 2)).astype(np.float32),
        'name': 'random_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 tensor with values very close to 0 and 1
    input_dict = {
        'x': np.array([1e-9, 1.0 - 1e-9, 1e-15, 1.0 - 1e-15], dtype=np.float64),
        'name': 'boundary_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tensor
    input_dict = {
        'x': np.array([], dtype=np.float32),
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with a single element
    input_dict = {
        'x': np.array([[[0.68]]], dtype=np.float64),
        'name': 'single_element'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large 1D float32 tensor
    input_dict = {
        'x': np.linspace(0.01, 0.99, 50, dtype=np.float32),
        'name': 'linspace_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 0-D (scalar) float64 tensor
    input_dict = {
        'x': np.array(0.975, dtype=np.float64),
        'name': 'scalar_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Values corresponding to standard deviations
    input_dict = {
        'x': np.array([0.15865525, 0.84134475, 0.02275013, 0.97724987], dtype=np.float32),
        'name': 'std_dev_probs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Ndtri"] = get_tf_raw_ops_ndtri_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Ndtri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Ndtri'.")

check_valid('tf.raw_ops.Ndtri', generated_inputs['tf.raw_ops.Ndtri'], lib="tf", suffix=0)
