
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_gradient_descent_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, 1D
    input_dict = {
        'var': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'alpha': np.array(0.1, dtype=np.float32),
        'delta': np.array([0.5, 0.4, 0.3], dtype=np.float32),
        'use_locking': False,
        'name': 'case1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D, with locking
    input_dict = {
        'var': np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        'alpha': np.array(0.01, dtype=np.float64),
        'delta': np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64),
        'use_locking': True,
        'name': 'case2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, scalar
    input_dict = {
        'var': np.array(100.0, dtype=np.float32),
        'alpha': np.array(0.5, dtype=np.float32),
        'delta': np.array(10.0, dtype=np.float32),
        'use_locking': False,
        'name': 'case3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 1D with negative values
    input_dict = {
        'var': np.array([-10.0, 20.0, -30.0], dtype=np.float64),
        'alpha': np.array(1.5, dtype=np.float64),
        'delta': np.array([-0.5, 0.4, 0.3], dtype=np.float64),
        'use_locking': True,
        'name': 'case4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D
    input_dict = {
        'var': np.ones((2, 2, 2), dtype=np.float32),
        'alpha': np.array(0.2, dtype=np.float32),
        'delta': np.full((2, 2, 2), 0.5, dtype=np.float32),
        'use_locking': False,
        'name': 'case5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Zero delta
    input_dict = {
        'var': np.array([10.0, 20.0], dtype=np.float32),
        'alpha': np.array(0.1, dtype=np.float32),
        'delta': np.zeros(2, dtype=np.float32),
        'use_locking': False,
        'name': 'case6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero alpha
    input_dict = {
        'var': np.array([[100.0]], dtype=np.float64),
        'alpha': np.array(0.0, dtype=np.float64),
        'delta': np.array([[50.0]], dtype=np.float64),
        'use_locking': True,
        'name': 'case7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D float32 with locking
    input_dict = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'alpha': np.array(1e-3, dtype=np.float32),
        'delta': np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        'use_locking': True,
        'name': 'case8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values
    input_dict = {
        'var': np.array([1e6, -1e6], dtype=np.float32),
        'alpha': np.array(0.001, dtype=np.float32),
        'delta': np.array([1e5, 1e5], dtype=np.float32),
        'use_locking': False,
        'name': 'case9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: small values
    input_dict = {
        'var': np.array([1e-6, -1e-6], dtype=np.float64),
        'alpha': np.array(0.001, dtype=np.float64),
        'delta': np.array([1e-5, -1e-5], dtype=np.float64),
        'use_locking': False,
        'name': 'case10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: half (float16)
    input_dict = {
        'var': np.array([1.0, 2.0], dtype=np.float16),
        'alpha': np.array(0.5, dtype=np.float16),
        'delta': np.array([0.1, 0.2], dtype=np.float16),
        'use_locking': False,
        'name': 'case11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyGradientDescent"] = get_apply_gradient_descent_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyGradientDescent'.")

check_valid('tf.raw_ops.ApplyGradientDescent', generated_inputs['tf.raw_ops.ApplyGradientDescent'], lib="tf", suffix=0)
