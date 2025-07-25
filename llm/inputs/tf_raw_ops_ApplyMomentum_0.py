
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_apply_momentum_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyMomentum.

    The persistent `RuntimeError: apply_momentum op does not support eager
    execution. Arg 'out' is a ref.` is a fundamental issue related to how
    TensorFlow has evolved. This specific raw operation (`ApplyMomentum`) is a
    "ref" operation, designed for the older TensorFlow graph mode where it
    modifies a `tf.Variable` in-place.

    Modern TensorFlow (TF2+) defaults to eager execution, which works with
    immutable `tf.Tensor` objects and uses a different system of "resource"
    variables (e.g., `tf.raw_ops.ResourceApplyMomentum`). The "ref" ops are not
    compatible with this eager execution model. The error is not caused by the
    values or dtypes of the numpy inputs, but by the attempt to run a
    graph-mode-only operation in an eager context.

    Since the problem lies in the execution environment's choice of op, no
    variation of the numpy inputs can resolve it. This response provides a
    single, canonical input that is perfectly valid according to the API's
    signature. The failure of this minimal case confirms the issue is with the
    execution context, not the input data.
    """
    list_of_inputs = []

    # A single, canonical example representing the most common use case.
    # This input is valid for the op's signature, even if the execution
    # mode is incompatible.
    input_dict = {
        'use_locking': False,
        'use_nesterov': False,
        'name': 'canonical_apply_momentum',
        'var': np.array([1.0, 2.0], dtype=np.float32),
        'accum': np.array([0.1, 0.2], dtype=np.float32),
        'lr': np.array(0.01, dtype=np.float32),
        'grad': np.array([0.5, -0.5], dtype=np.float32),
        'momentum': np.array(0.9, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyMomentum"] = get_apply_momentum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyMomentum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyMomentum'.")

check_valid('tf.raw_ops.ApplyMomentum', generated_inputs['tf.raw_ops.ApplyMomentum'], lib="tf", suffix=0)
