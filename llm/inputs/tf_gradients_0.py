
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# The user is encountering two alternating errors:
# 1. `AttributeError: 'list' object has no attribute 'shape'`
# 2. `RuntimeError: tf.gradients is not supported when eager execution is enabled.`
#
# The `AttributeError` occurs when `ys`, `xs`, etc., are provided as lists of numpy arrays,
# because the testing harness seems to expect a single object with a `.shape` attribute.
# The `RuntimeError` is fundamental because `tf.gradients` requires a graph context
# (i.e., to be run inside a `tf.function`), which the execution environment is not providing.
#
# To resolve the `AttributeError`, the following inputs provide single numpy arrays for
# `ys`, `xs`, `grad_ys`, and `stop_gradients`. According to the documentation, these
# parameters accept a single Tensor as well as a list of Tensors. This will satisfy
# the testing harness's immediate validation check. The `RuntimeError` will likely persist
# as it is an environmental issue beyond the scope of input generation.

def tf_gradients_inputs():
    list_of_inputs = []

    # Input 1: Basic unconnected case, requesting a zero gradient.
    input_dict_1 = {
        'ys': np.zeros((3, 3), dtype=np.float32),
        'xs': np.ones((2, 2), dtype=np.float32),
        'grad_ys': None,
        'name': 'unconnected_zero',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Unconnected case, requesting a None gradient (default).
    input_dict_2 = {
        'ys': np.array(-5.0, dtype=np.float32),
        'xs': np.array([10.0, 20.0], dtype=np.float32),
        'grad_ys': None,
        'name': 'unconnected_none',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using `grad_ys`. In a graph, this would scale the gradient.
    # We set xs=ys to simulate a scenario that would be connected.
    y3 = np.arange(6, dtype=np.float32).reshape(2, 3)
    grad_y3 = np.full((2, 3), 0.5, dtype=np.float32)
    input_dict_3 = {
        'ys': y3,
        'xs': y3,
        'grad_ys': grad_y3,
        'name': 'with_grad_ys',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using `stop_gradients` with float64.
    y4 = np.array(1.0, dtype=np.float64)
    input_dict_4 = {
        'ys': y4,
        'xs': y4,
        'grad_ys': None,
        'name': 'with_stop_gradients',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': y4,
        'unconnected_gradients': 'zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Using `gate_gradients=True`.
    y5 = np.array([[1.0]], dtype=np.float32)
    input_dict_5 = {
        'ys': y5,
        'xs': y5,
        'grad_ys': None,
        'name': 'with_gate_gradients',
        'gate_gradients': True,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Another unconnected case with different shapes.
    input_dict_6 = {
        'ys': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'xs': np.array(0.0, dtype=np.float32),
        'grad_ys': None,
        'name': 'vector_ys_scalar_xs',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: High dimensional unconnected tensors.
    input_dict_7 = {
        'ys': np.random.rand(2, 3, 4).astype(np.float32),
        'xs': np.random.rand(5, 6).astype(np.float32),
        'grad_ys': None,
        'name': 'high_dim_unconnected',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["tf.gradients"] = tf_gradients_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.gradients' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.gradients'.")

check_valid('tf.gradients', generated_inputs['tf.gradients'], lib="tf", suffix=0)
