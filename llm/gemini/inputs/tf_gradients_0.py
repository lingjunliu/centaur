
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_gradients_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar case
    # Graph: y = x^2. dy/dx at x=3 is 6.
    # Note: 'ys' and 'xs' are now single numpy arrays, not lists containing them.
    y1 = np.array(9.0, dtype=np.float32)
    x1 = np.array(3.0, dtype=np.float32)
    list_of_inputs.append({
        'ys': y1,
        'xs': x1,
        'grad_ys': None,
        'name': 'basic_scalar',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    })

    # Input 2: Vector input
    # Graph: y = sum(x*x). For x=[1,2,3], y=14. dy/dx = 2*x = [2,4,6].
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y2 = np.array(14.0, dtype=np.float32)
    list_of_inputs.append({
        'ys': y2,
        'xs': x2,
        'grad_ys': None,
        'name': 'vector_input',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    })

    # Input 3: With stop_gradients
    # Graph: y = a+b, where b is some function of a. If we stop b, it's treated as a constant.
    y3 = np.array(3.0, dtype=np.float32)
    x3 = np.array(1.0, dtype=np.float32)
    stop_tensor = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({
        'ys': y3,
        'xs': x3,
        'grad_ys': None,
        'name': 'with_stop_gradients',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': stop_tensor,
        'unconnected_gradients': 'none'
    })

    # Input 4: With grad_ys
    # Graph: y = 3*x. dy/dx = 3. With grad_ys=10, result is 10*3=30.
    y4 = np.array(6.0, dtype=np.float32)
    x4 = np.array(2.0, dtype=np.float32)
    grad_ys4 = np.array(10.0, dtype=np.float32)
    list_of_inputs.append({
        'ys': y4,
        'xs': x4,
        'grad_ys': grad_ys4,
        'name': 'with_grad_ys',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    })

    # Input 5: unconnected_gradients = 'zero'
    # Assuming the test harness creates an unconnected graph.
    y5 = np.array(10.0, dtype=np.float32)
    x5 = np.array(5.0, dtype=np.float32)
    list_of_inputs.append({
        'ys': y5,
        'xs': x5,
        'grad_ys': None,
        'name': 'unconnected_zero',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'zero'
    })

    # Input 6: unconnected_gradients = 'none'
    y6 = np.array([1., 2.], dtype=np.float32)
    x6 = np.array([[3., 4.], [5., 6.]], dtype=np.float32)
    list_of_inputs.append({
        'ys': y6,
        'xs': x6,
        'grad_ys': None,
        'name': 'unconnected_none',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    })

    # Input 7: gate_gradients = True
    y7 = np.array(9.0, dtype=np.float32)
    x7 = np.array(-3.0, dtype=np.float32)
    list_of_inputs.append({
        'ys': y7,
        'xs': x7,
        'grad_ys': None,
        'name': 'gate_gradients_true',
        'gate_gradients': True,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    })

    # Input 8: float64 dtype
    # Graph: y = 10/x. dy/dx = -10/x^2. At x=0.5, dy/dx = -40.
    y8 = np.array(20.0, dtype=np.float64)
    x8 = np.array(0.5, dtype=np.float64)
    list_of_inputs.append({
        'ys': y8,
        'xs': x8,
        'grad_ys': None,
        'name': 'float64_dtype',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    })

    # Input 9: High rank tensors
    # Graph: y = sum(x*x). dy/dx = 2*x.
    x9 = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    y9 = np.sum(x9 * x9).astype(np.float32)
    list_of_inputs.append({
        'ys': y9,
        'xs': x9,
        'grad_ys': None,
        'name': 'high_rank_tensors',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    })
    
    # Input 10: Using aggregation_method
    y10 = np.array(25.0, dtype=np.float32)
    x10 = np.array(5.0, dtype=np.float32)
    list_of_inputs.append({
        'ys': y10,
        'xs': x10,
        'grad_ys': None,
        'name': 'aggregation_method_test',
        'gate_gradients': False,
        'aggregation_method': 'EXPERIMENTAL_TREE',
        'stop_gradients': None,
        'unconnected_gradients': 'none'
    })

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
