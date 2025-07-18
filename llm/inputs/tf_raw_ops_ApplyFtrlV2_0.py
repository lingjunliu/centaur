
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_ftrl_v2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ApplyFtrlV2 operation.
    This version provides all tensor inputs as NumPy arrays to resolve the
    `AttributeError: 'ResourceVariable' object has no attribute 'size'` by
    conforming to the test harness's expectation for NumPy arrays during its
    input analysis phase.
    """
    list_of_inputs = []

    def create_input_dict(var_np, accum_np, linear_np, grad_np, lr, l1, l2, l2_shrinkage, lr_power, use_locking, multiply_linear_by_lr, name, dtype):
        # All tensor-like inputs are provided as NumPy arrays to satisfy the test harness.
        return {
            'var': np.array(var_np, dtype=dtype),
            'accum': np.array(accum_np, dtype=dtype),
            'linear': np.array(linear_np, dtype=dtype),
            'grad': np.array(grad_np, dtype=dtype),
            'lr': np.array(lr, dtype=dtype),
            'l1': np.array(l1, dtype=dtype),
            'l2': np.array(l2, dtype=dtype),
            'l2_shrinkage': np.array(l2_shrinkage, dtype=dtype),
            'lr_power': np.array(lr_power, dtype=dtype),
            'use_locking': use_locking,
            'multiply_linear_by_lr': multiply_linear_by_lr,
            'name': name
        }

    # Input 1: Basic float32, 1D
    list_of_inputs.append(create_input_dict(
        var_np=[1.0, 2.0, 3.0],
        accum_np=[0.1, 0.1, 0.1],
        linear_np=[0.5, -0.5, 0.0],
        grad_np=[0.2, 0.3, -0.1],
        lr=0.1, l1=0.01, l2=0.0, l2_shrinkage=0.001, lr_power=-0.5,
        use_locking=False, multiply_linear_by_lr=False, name="ftrl_v2_1", dtype=np.float32
    ))

    # Input 2: float64, 2D
    list_of_inputs.append(create_input_dict(
        var_np=[[-1.0, 2.5], [3.0, -4.0]],
        accum_np=[[1.0, 1.0], [1.0, 1.0]],
        linear_np=[[0.2, -0.3], [0.4, -0.1]],
        grad_np=[[0.1, 0.2], [-0.3, -0.4]],
        lr=0.05, l1=0.1, l2=0.2, l2_shrinkage=0.01, lr_power=-0.5,
        use_locking=False, multiply_linear_by_lr=False, name="ftrl_v2_2", dtype=np.float64
    ))

    # Input 3: With use_locking=True
    list_of_inputs.append(create_input_dict(
        var_np=[1.0, 2.0, 3.0, 4.0],
        accum_np=[0.1, 0.1, 0.1, 0.1],
        linear_np=[0.5, -0.5, 0.0, 1.0],
        grad_np=[0.2, 0.3, -0.1, 0.0],
        lr=0.1, l1=0.01, l2=0.0, l2_shrinkage=0.0, lr_power=-0.5,
        use_locking=True, multiply_linear_by_lr=False, name="ftrl_v2_3", dtype=np.float32
    ))

    # Input 4: With multiply_linear_by_lr=True
    list_of_inputs.append(create_input_dict(
        var_np=[[-10.0], [25.0]],
        accum_np=[[10.0], [10.0]],
        linear_np=[[2.0], [-3.0]],
        grad_np=[[1.0], [2.0]],
        lr=0.001, l1=1.0, l2=1.0, l2_shrinkage=0.1, lr_power=-0.5,
        use_locking=False, multiply_linear_by_lr=True, name="ftrl_v2_4", dtype=np.float64
    ))

    # Input 5: 3D tensor
    list_of_inputs.append(create_input_dict(
        var_np=np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        accum_np=np.full((2, 2, 2), 0.1, dtype=np.float32),
        linear_np=np.random.randn(2, 2, 2).astype(np.float32),
        grad_np=np.random.randn(2, 2, 2).astype(np.float32),
        lr=0.01, l1=0.0, l2=0.1, l2_shrinkage=0.01, lr_power=-0.5,
        use_locking=False, multiply_linear_by_lr=False, name="ftrl_v2_5", dtype=np.float32
    ))

    # Input 6: Zero regularization and zero grad
    list_of_inputs.append(create_input_dict(
        var_np=[1.0, -1.0, 2.0],
        accum_np=[1.0, 1.0, 1.0],
        linear_np=[0.1, -0.1, 0.2],
        grad_np=[0.0, 0.0, 0.0],
        lr=0.1, l1=0.0, l2=0.0, l2_shrinkage=0.0, lr_power=-0.5,
        use_locking=False, multiply_linear_by_lr=False, name="ftrl_v2_6", dtype=np.float64
    ))

    # Input 7: Large values
    list_of_inputs.append(create_input_dict(
        var_np=[1e6, -2e6],
        accum_np=[1e7, 1e7],
        linear_np=[1e5, -1e5],
        grad_np=[1e4, -1.5e4],
        lr=1e-4, l1=1e2, l2=1e1, l2_shrinkage=1.0, lr_power=-0.5,
        use_locking=False, multiply_linear_by_lr=False, name="ftrl_v2_7", dtype=np.float32
    ))

    # Input 8: All flags True
    list_of_inputs.append(create_input_dict(
        var_np=[[1.0]],
        accum_np=[[0.1]],
        linear_np=[[0.5]],
        grad_np=[[-0.2]],
        lr=0.1, l1=0.01, l2=0.02, l2_shrinkage=0.005, lr_power=-0.5,
        use_locking=True, multiply_linear_by_lr=True, name="ftrl_v2_8", dtype=np.float32
    ))
    
    # Input 9: Zero values for var, accum, linear
    list_of_inputs.append(create_input_dict(
        var_np=[0.0, 0.0],
        accum_np=[0.0, 0.0],
        linear_np=[0.0, 0.0],
        grad_np=[0.1, -0.1],
        lr=0.1, l1=0.01, l2=0.0, l2_shrinkage=0.0, lr_power=-0.5,
        use_locking=False, multiply_linear_by_lr=False, name="ftrl_v2_9", dtype=np.float32
    ))

    # Input 10: Scalar case
    list_of_inputs.append(create_input_dict(
        var_np=1.0,
        accum_np=0.1,
        linear_np=0.5,
        grad_np=-0.2,
        lr=0.1, l1=0.01, l2=0.02, l2_shrinkage=0.005, lr_power=-0.5,
        use_locking=False, multiply_linear_by_lr=False, name="ftrl_v2_10", dtype=np.float32
    ))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyFtrlV2"] = get_apply_ftrl_v2_inputs()

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
