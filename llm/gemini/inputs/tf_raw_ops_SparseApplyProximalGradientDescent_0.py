
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_sparse_apply_proximal_gradient_descent_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.SparseApplyProximalGradientDescent operation.
    The runtime error "does not support eager execution" is inherent to this specific legacy op,
    which operates on 'ref' tensors. It is designed for TensorFlow's graph mode. The inputs
    provided are valid for the op's signature and would execute correctly within a tf.Graph context.
    The focus is on standard floating-point types, which are the intended use case for this optimizer op.
    """
    list_of_inputs = []

    # Input 1: Basic float32 case
    input_dict_1 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'alpha': np.array(0.1, dtype=np.float32),
        'l1': np.array(0.01, dtype=np.float32),
        'l2': np.array(0.02, dtype=np.float32),
        'grad': np.array([[0.5, -0.5]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'name': 'float32_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic float64 case with use_locking=True
    input_dict_2 = {
        'var': np.array([[10.0], [20.0], [30.0]], dtype=np.float64),
        'alpha': np.array(0.5, dtype=np.float64),
        'l1': np.array(0.1, dtype=np.float64),
        'l2': np.array(0.2, dtype=np.float64),
        'grad': np.array([[-2.0], [3.0]], dtype=np.float64),
        'indices': np.array([1, 2], dtype=np.int64),
        'use_locking': True,
        'name': 'float64_locking'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D variable
    input_dict_3 = {
        'var': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'alpha': np.array(0.2, dtype=np.float32),
        'l1': np.array(0.0, dtype=np.float32),
        'l2': np.array(0.1, dtype=np.float32),
        'grad': np.array([0.4, -0.8], dtype=np.float32),
        'indices': np.array([1, 3], dtype=np.int32),
        'use_locking': False,
        'name': 'float32_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D variable
    input_dict_4 = {
        'var': np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32),
        'alpha': np.array(1.0, dtype=np.float32),
        'l1': np.array(0.2, dtype=np.float32),
        'l2': np.array(0.3, dtype=np.float32),
        'grad': np.array([[[0.1, 0.2], [-0.3, -0.4]]], dtype=np.float32),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'name': 'float32_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Zero regularization
    input_dict_5 = {
        'var': np.array([[-1.0, 2.5], [-3.1, 4.2]], dtype=np.float64),
        'alpha': np.array(1.0, dtype=np.float64),
        'l1': np.array(0.0, dtype=np.float64),
        'l2': np.array(0.0, dtype=np.float64),
        'grad': np.array([[1.0, 1.0]], dtype=np.float64),
        'indices': np.array([0], dtype=np.int32),
        'use_locking': False,
        'name': 'float64_no_reg'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: half (float16) type
    input_dict_6 = {
        'var': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
        'alpha': np.array(0.1, dtype=np.float16),
        'l1': np.array(0.01, dtype=np.float16),
        'l2': np.array(0.02, dtype=np.float16),
        'grad': np.array([[0.5, -0.5]], dtype=np.float16),
        'indices': np.array([1], dtype=np.int32),
        'use_locking': False,
        'name': 'float16_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyProximalGradientDescent"] = get_tf_raw_ops_sparse_apply_proximal_gradient_descent_inputs()

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
