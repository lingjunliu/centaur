
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_apply_adadelta_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyAdadelta.
    The recurring "eager execution" error suggests the issue is fundamental to
    how the test harness calls this type of 'ref' op. This attempt provides
    inputs of various dtypes listed in the documentation, including integer types,
    which are unusual for optimizers but technically valid according to the
    API signature. This is to test if different type dispatch paths in TensorFlow
    might avoid the error.
    """
    list_of_inputs = []

    def create_input(shape, dtype, use_locking, name):
        var = (np.random.rand(*shape) * 10).astype(dtype)
        accum = (np.random.rand(*shape) * 1).astype(dtype)
        accum_update = (np.random.rand(*shape) * 1).astype(dtype)
        grad = (np.random.rand(*shape) * 2 - 1).astype(dtype)

        if np.issubdtype(dtype, np.integer):
            lr = np.array(1, dtype=dtype)
            rho = np.array(1, dtype=dtype)
            epsilon = np.array(0, dtype=dtype)
        else:
            lr = np.array(0.001, dtype=dtype)
            rho = np.array(0.95, dtype=dtype)
            epsilon = np.array(1e-7, dtype=dtype)

        return {
            'var': var,
            'accum': accum,
            'accum_update': accum_update,
            'lr': lr,
            'rho': rho,
            'epsilon': epsilon,
            'grad': grad,
            'use_locking': use_locking,
            'name': name
        }

    # Input 1: Standard float32, 1D
    list_of_inputs.append(copy.deepcopy(create_input((4,), np.float32, False, "apply_adadelta_f32")))

    # Input 2: Standard float64, 2D, with locking
    list_of_inputs.append(copy.deepcopy(create_input((2, 2), np.float64, True, "apply_adadelta_f64")))
    
    # Input 3: int32, as per documentation
    list_of_inputs.append(copy.deepcopy(create_input((3,), np.int32, False, "apply_adadelta_i32")))

    # Input 4: int64, as per documentation
    list_of_inputs.append(copy.deepcopy(create_input((2, 3), np.int64, True, "apply_adadelta_i64")))

    # Input 5: uint8, as per documentation
    list_of_inputs.append(copy.deepcopy(create_input((5,), np.uint8, False, "apply_adadelta_ui8")))

    # Input 6: int16, as per documentation
    list_of_inputs.append(copy.deepcopy(create_input((10,), np.int16, True, "apply_adadelta_i16")))

    # Input 7: Scalar float32 (0-D tensor)
    list_of_inputs.append(copy.deepcopy(create_input((), np.float32, False, "apply_adadelta_scalar_f32")))
    
    # Input 8: Scalar int32 (0-D tensor)
    list_of_inputs.append(copy.deepcopy(create_input((), np.int32, True, "apply_adadelta_scalar_i32")))

    # Input 9: half (float16)
    f16_input = create_input((6,), np.float16, True, "apply_adadelta_f16")
    f16_input['epsilon'] = np.array(1e-4, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy(f16_input))
    
    # Input 10: 1-element vector, float32
    list_of_inputs.append(copy.deepcopy(create_input((1,), np.float32, False, "apply_adadelta_1_elem")))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdadelta"] = get_tf_raw_ops_apply_adadelta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdadelta'.")

check_valid('tf.raw_ops.ApplyAdadelta', generated_inputs['tf.raw_ops.ApplyAdadelta'], lib="tf", suffix=0)
