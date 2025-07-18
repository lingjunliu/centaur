
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_adagrad_da_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ApplyAdagradDA function.
    The inputs are provided in numpy format to be compatible with the testing harness
    which expects numpy-like objects (e.g., with a .size attribute).
    """
    list_of_inputs = []

    def create_input_dict(var_dtype, var_shape, use_locking, name):
        # Helper to create numpy arrays, handles scalar case correctly.
        def create_np_array(shape, scale=1.0, offset=0.0):
             if not shape:
                 return np.array(np.random.rand() * scale + offset)
             return (np.random.rand(*shape) * scale + offset)

        var = create_np_array(var_shape, 10, -5).astype(var_dtype)
        gradient_accumulator = create_np_array(var_shape).astype(var_dtype)
        gradient_squared_accumulator = np.abs(create_np_array(var_shape)).astype(var_dtype)
        grad = create_np_array(var_shape, 2, -1).astype(var_dtype)

        if np.iscomplexobj(var):
            var += (create_np_array(var_shape, 10, -5) * 1j).astype(var_dtype)
            gradient_accumulator += (create_np_array(var_shape) * 1j).astype(var_dtype)
            gradient_squared_accumulator = gradient_squared_accumulator.astype(var_dtype)
            grad += (create_np_array(var_shape, 2, -1) * 1j).astype(var_dtype)

        lr = np.array(0.01, dtype=var_dtype)
        l1 = np.array(0.1, dtype=var_dtype)
        l2 = np.array(0.2, dtype=var_dtype)
        if np.iscomplexobj(lr):
            lr, l1, l2 = lr + 0j, l1 + 0j, l2 + 0j

        global_step = np.array(np.random.randint(1, 1000), dtype=np.int64)

        return {
            'var': var,
            'gradient_accumulator': gradient_accumulator,
            'gradient_squared_accumulator': gradient_squared_accumulator,
            'grad': grad,
            'lr': lr,
            'l1': l1,
            'l2': l2,
            'global_step': global_step,
            'use_locking': use_locking,
            'name': name
        }

    # Input 1: Basic float32, 1D
    list_of_inputs.append(create_input_dict(np.float32, (10,), False, 'test1_float32_1d'))

    # Input 2: float64, 2D with locking
    list_of_inputs.append(create_input_dict(np.float64, (3, 3), True, 'test2_float64_2d'))

    # Input 3: half (float16), 3D
    list_of_inputs.append(create_input_dict(np.float16, (2, 3, 4), False, 'test3_float16_3d'))

    # Input 4: Scalar variable
    list_of_inputs.append(create_input_dict(np.float32, (), False, 'test4_scalar'))

    # Input 5: Zero-valued gradients and accumulators
    input_5 = create_input_dict(np.float32, (4,), False, 'test5_zeros')
    input_5['gradient_accumulator'].fill(0)
    input_5['gradient_squared_accumulator'].fill(0)
    input_5['grad'].fill(0)
    list_of_inputs.append(input_5)

    # Input 6: Negative values in var and grad
    input_6 = create_input_dict(np.float64, (2, 5), True, 'test6_negatives')
    input_6['var'] = -np.abs(input_6['var'])
    input_6['grad'] = -np.abs(input_6['grad'])
    list_of_inputs.append(input_6)

    # Input 7: High L1/L2 regularization
    input_7 = create_input_dict(np.float32, (8,), False, 'test7_high_reg')
    input_7['l1'] = np.array(100.0, dtype=np.float32)
    input_7['l2'] = np.array(100.0, dtype=np.float32)
    list_of_inputs.append(input_7)

    # Input 8: Large global step
    input_8 = create_input_dict(np.float64, (6,), True, 'test8_large_step')
    input_8['global_step'] = np.array(999999999, dtype=np.int64)
    list_of_inputs.append(input_8)

    # Input 9: complex64 type
    list_of_inputs.append(create_input_dict(np.complex64, (2, 2), False, 'test9_complex64'))

    # Input 10: complex128 type
    list_of_inputs.append(create_input_dict(np.complex128, (4,), True, 'test10_complex128'))

    return [copy.deepcopy(i) for i in list_of_inputs]

generated_inputs["tf.raw_ops.ApplyAdagradDA"] = get_apply_adagrad_da_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdagradDA' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagradDA'.")

check_valid('tf.raw_ops.ApplyAdagradDA', generated_inputs['tf.raw_ops.ApplyAdagradDA'], lib="tf", suffix=0)
