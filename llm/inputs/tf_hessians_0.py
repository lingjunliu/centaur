
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_hessians_inputs():
    """
    Generates a list of syntactically valid inputs for the tf.hessians function.
    NOTE: tf.hessians is a TF1 compatibility API and is expected to raise a
    RuntimeError if called in eager mode (default in TF2). These inputs are
    correct for the API signature, assuming a TF1 graph context.
    The 'ys' and 'xs' parameters accept a single tensor as well as a list,
    so a single numpy array is provided.
    """
    list_of_inputs = []

    # Input 1: Basic case with a 1D float32 tensor for xs.
    list_of_inputs.append({
        'ys': np.array(0.0, dtype=np.float32),
        'xs': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessians_1d'
    })

    # Input 2: With a 2D tensor for xs.
    list_of_inputs.append({
        'ys': np.array(0.0, dtype=np.float32),
        'xs': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessians_2d'
    })

    # Input 3: Scalar input for xs.
    list_of_inputs.append({
        'ys': np.array(0.0, dtype=np.float32),
        'xs': np.array(5.0, dtype=np.float32),
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessians_scalar_xs'
    })

    # Input 4: With ys as a vector. The Hessian is of sum(ys), which is scalar.
    list_of_inputs.append({
        'ys': np.array([1.0, 2.0], dtype=np.float32),
        'xs': np.array([3.0, 4.0], dtype=np.float32),
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessians_vector_ys'
    })

    # Input 5: Edge case with an empty tensor for xs. This should return an empty list.
    list_of_inputs.append({
        'ys': np.array(0.0, dtype=np.float32),
        'xs': np.array([], dtype=np.float32),
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessians_empty_xs'
    })

    # Input 6: Edge case with an empty tensor for ys. sum(ys) is 0.
    list_of_inputs.append({
        'ys': np.array([], dtype=np.float32),
        'xs': np.array([1.0, 2.0], dtype=np.float32),
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessians_empty_ys'
    })
    
    # Input 7: Using float64 dtype.
    list_of_inputs.append({
        'ys': np.array(0.0, dtype=np.float64),
        'xs': np.array([-1.0, 0.0], dtype=np.float64),
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessians_float64'
    })
    
    # Input 8: Both ys and xs are scalars.
    list_of_inputs.append({
        'ys': np.array(10.0, dtype=np.float32),
        'xs': np.array(-5.0, dtype=np.float32),
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessians_scalar_both'
    })

    return list_of_inputs

generated_inputs["tf.hessians"] = get_tf_hessians_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.hessians' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.hessians'.")

check_valid('tf.hessians', generated_inputs['tf.hessians'], lib="tf", suffix=0)
