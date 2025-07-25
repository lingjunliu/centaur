
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_apply_proximal_gradient_descent_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyProximalGradientDescent.
    The 'var' input is a tf.Variable to handle the in-place update, which
    is required to avoid the eager execution error for 'ref' arguments. Other
    tensor inputs remain as numpy arrays.
    """
    list_of_inputs = []

    # Input 1: Basic float32 case
    input_dict = {
        'var': tf.Variable(np.array([1.0, 2.0, -3.0], dtype=np.float32)),
        'alpha': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.2, dtype=np.float32),
        'l2': np.array(0.01, dtype=np.float32),
        'delta': np.array([0.5, -0.5, 0.1], dtype=np.float32),
        'use_locking': False,
        'name': "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 case with 2D tensor and locking
    input_dict = {
        'var': tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)),
        'alpha': np.array(0.05, dtype=np.float64),
        'l1': np.array(1.0, dtype=np.float64),
        'l2': np.array(0.5, dtype=np.float64),
        'delta': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        'use_locking': True,
        'name': "test_float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: No L1 regularization
    input_dict = {
        'var': tf.Variable(np.array([[1.5, -2.5], [0.0, 5.0]], dtype=np.float32)),
        'alpha': np.array(0.2, dtype=np.float32),
        'l1': np.array(0.0, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'delta': np.array([[0.2, -0.3], [0.1, -0.4]], dtype=np.float32),
        'use_locking': False,
        'name': "no_l1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: No L2 regularization
    input_dict = {
        'var': tf.Variable(np.array([[1.5, -2.5]], dtype=np.float32)),
        'alpha': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.5, dtype=np.float32),
        'l2': np.array(0.0, dtype=np.float32),
        'delta': np.array([[0.2, -0.3]], dtype=np.float32),
        'use_locking': True,
        'name': "no_l2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: No regularization at all
    input_dict = {
        'var': tf.Variable(np.array([10.0, 20.0, 30.0], dtype=np.float64)),
        'alpha': np.array(0.01, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(0.0, dtype=np.float64),
        'delta': np.array([-1.0, 1.0, -1.0], dtype=np.float64),
        'use_locking': False,
        'name': "no_regularization"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor
    input_dict = {
        'var': tf.Variable(np.arange(8, dtype=np.float32).reshape(2, 2, 2)),
        'alpha': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.1, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'delta': np.ones((2, 2, 2), dtype=np.float32) * 0.5,
        'use_locking': True,
        'name': "3d_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero delta (update only from regularization)
    input_dict = {
        'var': tf.Variable(np.array([5.0, -5.0], dtype=np.float32)),
        'alpha': np.array(0.1, dtype=np.float32),
        'l1': np.array(1.0, dtype=np.float32),
        'l2': np.array(0.2, dtype=np.float32),
        'delta': np.array([0.0, 0.0], dtype=np.float32),
        'use_locking': False,
        'name': "zero_delta"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High regularization, should shrink var towards zero
    input_dict = {
        'var': tf.Variable(np.array([100.0, -200.0, 50.0], dtype=np.float32)),
        'alpha': np.array(0.1, dtype=np.float32),
        'l1': np.array(50.0, dtype=np.float32),
        'l2': np.array(10.0, dtype=np.float32),
        'delta': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'use_locking': False,
        'name': "high_reg"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half (float16) type
    input_dict = {
        'var': tf.Variable(np.array([1.0, 2.0], dtype=np.float16)),
        'alpha': np.array(0.1, dtype=np.float16),
        'l1': np.array(0.2, dtype=np.float16),
        'l2': np.array(0.01, dtype=np.float16),
        'delta': np.array([0.5, -0.5], dtype=np.float16),
        'use_locking': False,
        'name': 'float16_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: another float16 case
    input_dict = {
        'var': tf.Variable(np.array([-10.5, 0.0, 10.5], dtype=np.float16)),
        'alpha': np.array(0.01, dtype=np.float16),
        'l1': np.array(1.0, dtype=np.float16),
        'l2': np.array(0.1, dtype=np.float16),
        'delta': np.array([1.0, 1.0, -1.0], dtype=np.float16),
        'use_locking': True,
        'name': 'float16_test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyProximalGradientDescent"] = tf_raw_ops_apply_proximal_gradient_descent_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyProximalGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyProximalGradientDescent'.")

check_valid('tf.raw_ops.ApplyProximalGradientDescent', generated_inputs['tf.raw_ops.ApplyProximalGradientDescent'], lib="tf", suffix=0)
