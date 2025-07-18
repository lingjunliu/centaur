
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_accumulator_apply_gradient_inputs():
    """
    Generates a list of syntactically valid inputs for the tf.raw_ops.AccumulatorApplyGradient operation.

    NOTE: This operation is stateful and designed for TensorFlow's graph mode. It modifies
    a resource ('handle') that must be created by another operation (e.g., AccumulatorV2).
    The testing environment appears to use Eager Execution, where operations run immediately.
    Stateful resource operations like this one are incompatible with eager execution and will
    consistently raise a `RuntimeError: ... op does not support eager execution. ...`.
    The error is fundamental to the API's design and the execution context, not the inputs themselves.
    The following inputs are provided to satisfy the generation requirement, but they are expected
    to fail at runtime with the aforementioned error.
    """
    list_of_inputs = []

    def create_input_dict(handle_id, local_step, gradient, name=None):
        # The 'handle' is a reference to a resource. In numpy representation for eager
        # execution, a string in an object array is a placeholder.
        return {
            'handle': np.array([handle_id], dtype=object),
            'local_step': np.array(local_step, dtype=np.int64),
            'gradient': gradient,
            'name': name
        }

    # Input 1: Basic float32 gradient
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            'handle_1', 10,
            np.array([1.0, 2.5, -3.0], dtype=np.float32),
            name="apply_grad_float32"
        )
    ))

    # Input 2: int32 gradient, 2D
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            'handle_2', 100,
            np.array([[-1, 2], [3, -4]], dtype=np.int32)
        )
    ))

    # Input 3: complex64 gradient
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            'handle_3', 50,
            np.array([1+2j, 3-4j, -5+6j], dtype=np.complex64),
            name="apply_grad_complex64"
        )
    ))

    # Input 4: half (float16) gradient, scalar
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            'handle_4', 42,
            np.array(3.5, dtype=np.float16),
            name="apply_grad_half"
        )
    ))

    # Input 5: uint8 gradient, 3D
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            'handle_5', 20,
            np.arange(8, dtype=np.uint8).reshape((2, 2, 2))
        )
    ))

    return list_of_inputs

generated_inputs["tf.raw_ops.AccumulatorApplyGradient"] = tf_raw_ops_accumulator_apply_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorApplyGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorApplyGradient'.")

check_valid('tf.raw_ops.AccumulatorApplyGradient', generated_inputs['tf.raw_ops.AccumulatorApplyGradient'], lib="tf", suffix=0)
