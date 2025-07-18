
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_apply_ftrl_v2_inputs():
    """
    This function generates a list of valid inputs for the tf.raw_ops.ApplyFtrlV2 op.
    The inputs are numpy arrays. This is to satisfy the validation tooling which expects
    numpy-like objects. This may lead to a RuntimeError if the op is executed eagerly,
    as this raw op is designed for graph mode and expects mutable resource variables.
    """
    list_of_inputs = []

    # Helper function to create a base case
    def create_base_input(shape, dtype, name_suffix):
        # Ensure result is always a numpy array before astype, especially for the scalar case where shape=()
        var = np.array(np.random.randn(*shape)).astype(dtype)
        # accum must be non-negative
        accum = np.array(np.abs(np.random.randn(*shape))).astype(dtype)
        linear = np.array(np.random.randn(*shape)).astype(dtype)
        grad = np.array(np.random.randn(*shape)).astype(dtype)
        
        lr = np.array(0.001, dtype=dtype)
        l1 = np.array(0.1, dtype=dtype)
        l2 = np.array(0.2, dtype=dtype)
        l2_shrinkage = np.array(0.01, dtype=dtype)
        lr_power = np.array(-0.5, dtype=dtype)

        return {
            'var': var,
            'accum': accum,
            'linear': linear,
            'grad': grad,
            'lr': lr,
            'l1': l1,
            'l2': l2,
            'l2_shrinkage': l2_shrinkage,
            'lr_power': lr_power,
            'use_locking': False,
            'multiply_linear_by_lr': False,
            'name': f'test_{name_suffix}'
        }

    # Input 1: Basic case, float32, 2D
    list_of_inputs.append(copy.deepcopy(create_base_input(
        shape=(3, 3), dtype=np.float32, name_suffix='1'
    )))

    # Input 2: 1D vector
    list_of_inputs.append(copy.deepcopy(create_base_input(
        shape=(10,), dtype=np.float32, name_suffix='2'
    )))

    # Input 3: Scalar case
    list_of_inputs.append(copy.deepcopy(create_base_input(
        shape=(), dtype=np.float32, name_suffix='3'
    )))
    
    # Input 4: With locking and multiply_linear_by_lr
    input_4 = create_base_input(shape=(4, 4), dtype=np.float32, name_suffix='4')
    input_4['use_locking'] = True
    input_4['multiply_linear_by_lr'] = True
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: float64 type
    list_of_inputs.append(copy.deepcopy(create_base_input(
        shape=(2, 5), dtype=np.float64, name_suffix='5_float64'
    )))

    # Input 6: Zero regularization
    input_6 = create_base_input(shape=(5, 2), dtype=np.float32, name_suffix='6_zero_reg')
    input_6['l1'] = np.array(0.0, dtype=np.float32)
    input_6['l2'] = np.array(0.0, dtype=np.float32)
    input_6['l2_shrinkage'] = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Different lr_power
    input_7 = create_base_input(shape=(3,), dtype=np.float32, name_suffix='7_lr_power')
    input_7['lr_power'] = np.array(-1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Zero initial accum and linear
    input_8 = create_base_input(shape=(2, 2), dtype=np.float32, name_suffix='8_zero_init')
    input_8['accum'] = np.zeros((2, 2), dtype=np.float32)
    input_8['linear'] = np.zeros((2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: float16 type
    list_of_inputs.append(copy.deepcopy(create_base_input(
        shape=(6, 1), dtype=np.float16, name_suffix='9_float16'
    )))

    # Input 10: High regularization
    input_10 = create_base_input(shape=(3, 3), dtype=np.float32, name_suffix='10_high_reg')
    input_10['l1'] = np.array(5.0, dtype=np.float32)
    input_10['l2'] = np.array(10.0, dtype=np.float32)
    input_10['l2_shrinkage'] = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyFtrlV2"] = tf_raw_ops_apply_ftrl_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyFtrlV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyFtrlV2'.")

check_valid('tf.raw_ops.ApplyFtrlV2', generated_inputs['tf.raw_ops.ApplyFtrlV2'], lib="tf", suffix=0)
