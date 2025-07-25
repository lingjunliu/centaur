
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_sparse_apply_momentum_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.SparseApplyMomentum function.

    **NOTE ON EXECUTION_MODE:** The target operation `tf.raw_ops.SparseApplyMomentum`
    is a legacy TensorFlow 1.x graph-mode operation. It is explicitly disabled
    in TensorFlow's eager execution mode, which is the default in modern
    TensorFlow. Any attempt to call this function directly while in eager mode
    will deterministically raise a `RuntimeError`.

    The modern, eager-compatible equivalent is `tf.raw_ops.ResourceSparseApplyMomentum`.

    Because the testing harness calls the specified function in eager mode, it is
    impossible to generate an input that will pass execution for the requested
    `tf.raw_ops.SparseApplyMomentum` op. The error is fundamental to the
    operation's design and its incompatibility with the execution context, not
    the input values themselves.

    Therefore, an empty list of inputs is being returned to signify that no valid
    execution path exists under the given constraints.
    """
    list_of_inputs = []
    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyMomentum"] = tf_raw_ops_sparse_apply_momentum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyMomentum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyMomentum'.")

check_valid('tf.raw_ops.SparseApplyMomentum', generated_inputs['tf.raw_ops.SparseApplyMomentum'], lib="tf", suffix=0)
