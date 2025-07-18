
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_centered_rmsprop_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyCenteredRMSProp.
    """
    list_of_inputs = []

    def _generate_input(dtype, shape, use_locking, name, grad_is_zero=False, custom_params=None):
        params = {'lr': 0.001, 'rho': 0.9, 'momentum': 0.5, 'epsilon': 1e-7}
        if custom_params:
            params.update(custom_params)

        var = (np.random.uniform(size=shape) * 20 - 10).astype(dtype)
        mg = (np.random.uniform(size=shape) * 4 - 2).astype(dtype)
        ms = np.square(mg) + (np.random.uniform(size=shape) * 5).astype(dtype)
        mom = (np.random.uniform(size=shape) * 20 - 10).astype(dtype)
        grad = np.zeros(shape, dtype=dtype) if grad_is_zero else (np.random.uniform(size=shape) * 4 - 2).astype(dtype)
        
        input_dict = {
            'use_locking': use_locking,
            'name': name,
            'var': var,
            'mg': mg,
            'ms': ms,
            'mom': mom,
            'grad': grad,
            'lr': np.array(params['lr'], dtype=dtype),
            'rho': np.array(params['rho'], dtype=dtype),
            'momentum': np.array(params['momentum'], dtype=dtype),
            'epsilon': np.array(params['epsilon'], dtype=dtype)
        }
        return input_dict
    
    def _generate_complex_input(dtype, shape, use_locking, name):
        var = (np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)).astype(dtype)
        mg = (np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)).astype(dtype)
        ms = (np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)).astype(dtype)
        mom = (np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)).astype(dtype)
        grad = (np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)).astype(dtype)
        params = {'lr': 0.001, 'rho': 0.9, 'momentum': 0.5, 'epsilon': 1e-7}

        input_dict = {
            'use_locking': use_locking, 'name': name,
            'var': var, 'mg': mg, 'ms': ms, 'mom': mom, 'grad': grad,
            'lr': np.array(params['lr'], dtype=dtype),
            'rho': np.array(params['rho'], dtype=dtype),
            'momentum': np.array(params['momentum'], dtype=dtype),
            'epsilon': np.array(params['epsilon'], dtype=dtype)
        }
        return input_dict


    list_of_inputs.append(copy.deepcopy(_generate_input(np.float32, (3, 3), False, "case1_float32_2d")))
    list_of_inputs.append(copy.deepcopy(_generate_input(np.float64, (10,), True, "case2_float64_1d_locked")))
    list_of_inputs.append(copy.deepcopy(_generate_input(np.float32, (4, 2), False, "case3_zero_grad", grad_is_zero=True)))
    list_of_inputs.append(copy.deepcopy(_generate_input(np.float32, (), False, "case4_scalar_float32")))
    list_of_inputs.append(copy.deepcopy(_generate_input(np.float32, (2, 2), False, "case5_large_epsilon", custom_params={'epsilon': 1.5})))
    list_of_inputs.append(copy.deepcopy(_generate_input(np.float64, (5, 2), False, "case6_custom_hyperparams", custom_params={'lr': 0.1, 'momentum': 0.99, 'rho': 0.95})))
    list_of_inputs.append(copy.deepcopy(_generate_input(np.float32, (3, 3), False, "case7_neg_momentum", custom_params={'momentum': -0.5})))
    list_of_inputs.append(copy.deepcopy(_generate_input(np.float64, (2,2,2), False, "case8_float64_3d")))
    list_of_inputs.append(copy.deepcopy(_generate_complex_input(np.complex64, (2, 4), False, "case9_complex64")))
    list_of_inputs.append(copy.deepcopy(_generate_complex_input(np.complex128, (6,), True, "case10_complex128")))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyCenteredRMSProp"] = get_apply_centered_rmsprop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyCenteredRMSProp'.")

check_valid('tf.raw_ops.ApplyCenteredRMSProp', generated_inputs['tf.raw_ops.ApplyCenteredRMSProp'], lib="tf", suffix=0)
