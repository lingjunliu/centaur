
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_sparse_apply_rmsprop_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.SparseApplyRMSProp function.

    The error "AttributeError: 'ResourceVariable' object has no attribute 'size'"
    occurs because the testing framework expects numpy arrays (or objects with a .size
    attribute) for tensor-like inputs, but a previous fix to address a TensorFlow
    runtime error involved converting these inputs to tf.Variable objects.

    This implementation reverts to providing numpy arrays for all tensor inputs to
    resolve the AttributeError in the testing framework's input analysis stage.
    This will likely re-introduce the "does not support eager execution" RuntimeError
    during the execution stage, as this specific op requires mutable tf.Variable
    inputs. This indicates a fundamental conflict between the testing framework's
    requirements and the API's requirements for stateful operations.
    """
    list_of_inputs = []

    # Helper function to create a single input dictionary with numpy arrays.
    def create_input_dict(var, ms, mom, lr, rho, momentum, epsilon, grad, indices, use_locking, name):
        dtype = var.dtype
        return {
            'var': var,
            'ms': ms,
            'mom': mom,
            'lr': np.array(lr, dtype=dtype),
            'rho': np.array(rho, dtype=dtype),
            'momentum': np.array(momentum, dtype=dtype),
            'epsilon': np.array(epsilon, dtype=dtype),
            'grad': grad,
            'indices': indices,
            'use_locking': use_locking,
            'name': name
        }

    # Input 1: Basic float32 case
    var1 = np.arange(15, dtype=np.float32).reshape(5, 3)
    ms1 = np.ones((5, 3), dtype=np.float32)
    mom1 = np.zeros((5, 3), dtype=np.float32)
    grad1 = np.random.randn(2, 3).astype(np.float32)
    indices1 = np.array([0, 3], dtype=np.int32)
    list_of_inputs.append(create_input_dict(
        var1, ms1, mom1, 0.01, 0.9, 0.0, 1e-7, grad1, indices1, False, "basic_float32"
    ))

    # Input 2: float64 with locking enabled
    var2 = np.random.randn(10, 2).astype(np.float64)
    ms2 = np.ones((10, 2), dtype=np.float64)
    mom2 = np.zeros((10, 2), dtype=np.float64)
    grad2 = np.random.randn(3, 2).astype(np.float64)
    indices2 = np.array([1, 8, 5], dtype=np.int64)
    list_of_inputs.append(create_input_dict(
        var2, ms2, mom2, 0.001, 0.99, 0.1, 1e-8, grad2, indices2, True, "float64_with_locking"
    ))

    # Input 3: Higher dimensional tensor (3D)
    var3 = np.random.rand(4, 3, 2).astype(np.float32)
    ms3 = np.ones((4, 3, 2), dtype=np.float32)
    mom3 = np.zeros((4, 3, 2), dtype=np.float32)
    grad3 = np.random.rand(2, 3, 2).astype(np.float32)
    indices3 = np.array([0, 2], dtype=np.int32)
    list_of_inputs.append(create_input_dict(
        var3, ms3, mom3, 0.1, 0.8, 0.5, 1e-6, grad3, indices3, False, "higher_dims_float32"
    ))

    # Input 4: Empty indices (no-op)
    var4 = np.arange(10, dtype=np.float32).reshape(5, 2)
    ms4 = np.ones((5, 2), dtype=np.float32)
    mom4 = np.zeros((5, 2), dtype=np.float32)
    grad4 = np.empty((0, 2), dtype=np.float32)
    indices4 = np.array([], dtype=np.int32)
    list_of_inputs.append(create_input_dict(
        var4, ms4, mom4, 0.01, 0.9, 0.0, 1e-7, grad4, indices4, False, "empty_indices"
    ))

    # Input 5: Duplicate indices
    var5 = np.zeros((5, 2), dtype=np.float32)
    ms5 = np.ones((5, 2), dtype=np.float32)
    mom5 = np.zeros((5, 2), dtype=np.float32)
    grad5 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    indices5 = np.array([1, 4, 1], dtype=np.int32)
    list_of_inputs.append(create_input_dict(
        var5, ms5, mom5, 1.0, 0.9, 0.0, 1e-7, grad5, indices5, False, "duplicate_indices"
    ))

    # Input 6: Zero gradient
    var6 = np.ones((5, 5), dtype=np.float32)
    ms6 = np.ones((5, 5), dtype=np.float32)
    mom6 = np.ones((5, 5), dtype=np.float32)
    grad6 = np.zeros((2, 5), dtype=np.float32)
    indices6 = np.array([2, 4], dtype=np.int32)
    list_of_inputs.append(create_input_dict(
        var6, ms6, mom6, 0.1, 0.9, 0.9, 1e-7, grad6, indices6, False, "zero_gradient"
    ))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyRMSProp"] = tf_raw_ops_sparse_apply_rmsprop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyRMSProp'.")

check_valid('tf.raw_ops.SparseApplyRMSProp', generated_inputs['tf.raw_ops.SparseApplyRMSProp'], lib="tf", suffix=0)
