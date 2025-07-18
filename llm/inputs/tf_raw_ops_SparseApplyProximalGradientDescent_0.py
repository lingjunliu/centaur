
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_SparseApplyProximalGradientDescent_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseApplyProximalGradientDescent.
    The RuntimeError "does not support eager execution" is fundamental to this raw op,
    as it's designed to modify a tf.Variable in-place within a TensorFlow graph.
    The provided input is valid for such a graph-based context.
    """
    list_of_inputs = []

    # A single, minimal, canonical example for a graph-based execution context.
    # This op is not expected to work in eager mode, which is the cause of the persistent error.
    input_dict_1 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'alpha': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.01, dtype=np.float32),
        'l2': np.array(0.02, dtype=np.float32),
        'grad': np.array([[0.5, 0.6]], dtype=np.float32),
        'indices': np.array([1], dtype=np.int32),
        'use_locking': False,
        'name': 'graph_only_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyProximalGradientDescent"] = tf_raw_ops_SparseApplyProximalGradientDescent_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyProximalGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyProximalGradientDescent'.")

check_valid('tf.raw_ops.SparseApplyProximalGradientDescent', generated_inputs['tf.raw_ops.SparseApplyProximalGradientDescent'], lib="tf", suffix=0)
