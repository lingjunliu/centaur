
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

# Helper class to bridge the gap between tf.Variable and a testing harness
# that expects numpy-like attributes (.size) and behavior (np.min/max).
class PatchedVariable:
    """
    A wrapper around tf.Variable that provides a `.size` attribute and
    supports np.min/np.max for compatibility with a testing harness,
    while still functioning as a tf.Variable for TensorFlow ops.
    It also correctly handles deepcopying.
    """
    def __init__(self, initial_value):
        # initial_value is expected to be a numpy array
        self._variable = tf.Variable(initial_value)
        self._initial_value_np = np.array(initial_value)

    @property
    def size(self):
        return self._initial_value_np.size

    # This allows the object to be converted to a numpy array,
    # which lets np.min, np.max, etc., work on it via np.asanyarray.
    def __array__(self, dtype=None):
        return self._variable.numpy()

    # Delegate all other attribute access to the underlying tf.Variable
    def __getattr__(self, name):
        return getattr(self._variable, name)

    # Custom deepcopy implementation to ensure the wrapper and the
    # underlying variable state are copied correctly.
    def __deepcopy__(self, memo):
        # Create a new PatchedVariable with a deep-copied numpy array
        new_initial_value = copy.deepcopy(self._initial_value_np, memo)
        new_instance = PatchedVariable(new_initial_value)
        memo[id(self)] = new_instance
        return new_instance

def get_tf_raw_ops_apply_ftrl_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ApplyFtrl function.
    """
    list_of_inputs = []

    def _create_input_dict(dtype, shape, use_locking, multiply_linear_by_lr, name_suffix):
        var_np = (np.random.rand(*shape) * 10).astype(dtype)
        # accum must be non-negative for the formula to be well-defined
        accum_np = (np.random.rand(*shape) * 2 + 0.1).astype(dtype)
        linear_np = (np.random.randn(*shape)).astype(dtype)
        
        # grad and scalar params are not modified in-place and can be standard numpy arrays
        grad_val = (np.random.randn(*shape)).astype(dtype)
        lr_val = np.array(0.01, dtype=dtype)
        l1_val = np.array(0.1, dtype=dtype)
        l2_val = np.array(0.05, dtype=dtype)
        # lr_power must be <= 0 according to the FTRL algorithm
        lr_power_val = np.array(-0.5, dtype=dtype)

        # Use the PatchedVariable wrapper for mutable inputs
        input_dict = {
            'var': PatchedVariable(var_np),
            'accum': PatchedVariable(accum_np),
            'linear': PatchedVariable(linear_np),
            'grad': grad_val,
            'lr': lr_val,
            'l1': l1_val,
            'l2': l2_val,
            'lr_power': lr_power_val,
            'use_locking': use_locking,
            'multiply_linear_by_lr': multiply_linear_by_lr,
            'name': f"test_{name_suffix}"
        }
        return input_dict

    # Input 1: Basic float32, 1D
    list_of_inputs.append(copy.deepcopy(_create_input_dict(np.float32, (3,), False, False, "float32_1d")))

    # Input 2: float64, 2D, with locking and multiply_linear_by_lr
    list_of_inputs.append(copy.deepcopy(_create_input_dict(np.float64, (2, 2), True, True, "float64_2d")))

    # Input 3: No L1/L2 regularization
    input_3 = _create_input_dict(np.float32, (5,), True, False, "no_reg")
    input_3['l1'] = np.array(0.0, dtype=np.float32)
    input_3['l2'] = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Different lr_power
    input_4 = _create_input_dict(np.float64, (2, 3), False, False, "diff_lr_power")
    input_4['lr_power'] = np.array(-0.8, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy(input_4))
    
    # Input 5: Large gradient values
    input_5 = _create_input_dict(np.float32, (10,), False, False, "large_grad")
    input_5['grad'] = (np.random.rand(10) * 100 - 50).astype(np.float32)
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: Zero initial accumulator and linear
    input_6 = _create_input_dict(np.float32, (2, 2, 2), True, True, "zero_accum_linear")
    input_6['accum'] = PatchedVariable(np.zeros((2, 2, 2), dtype=np.float32))
    input_6['linear'] = PatchedVariable(np.zeros((2, 2, 2), dtype=np.float32))
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: High L1 regularization to trigger the `else 0.0` path
    input_7 = _create_input_dict(np.float32, (6,), False, True, "high_l1")
    input_7['l1'] = np.array(100.0, dtype=np.float32) 
    input_7['linear'] = PatchedVariable((np.random.rand(6) * 10).astype(np.float32))
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: 3D tensor
    list_of_inputs.append(copy.deepcopy(_create_input_dict(np.float32, (2, 3, 4), False, False, "float32_3d")))
    
    # Input 9: A different set of parameters
    input_9 = _create_input_dict(np.float32, (5,), False, True, "diff_params")
    input_9['lr'] = np.array(0.1, dtype=np.float32)
    input_9['l1'] = np.array(1.0, dtype=np.float32)
    input_9['l2'] = np.array(0.5, dtype=np.float32)
    input_9['lr_power'] = np.array(-0.1, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: Another float64 case
    list_of_inputs.append(copy.deepcopy(_create_input_dict(np.float64, (10,), True, False, "float64_large")))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyFtrl"] = get_tf_raw_ops_apply_ftrl_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyFtrl' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyFtrl'.")

check_valid('tf.raw_ops.ApplyFtrl', generated_inputs['tf.raw_ops.ApplyFtrl'], lib="tf", suffix=0)
