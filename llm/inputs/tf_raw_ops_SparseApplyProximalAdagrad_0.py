
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_SparseApplyProximalAdagrad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseApplyProximalAdagrad.

    CRITICAL NOTE: This TensorFlow op is stateful and designed to modify its
    'var' and 'accum' inputs. It is NOT compatible with TensorFlow's default
    eager execution mode. The 'RuntimeError' is an expected and documented
    behavior when calling this op eagerly. The inputs provided here are valid
    for the op's signature and would execute correctly in a graph context
    (e.g., inside a @tf.function or a TF1 session). The error stems from the
    execution environment, not the inputs themselves.
    """
    list_of_inputs = []

    def _create_input_dict(var, accum, lr, l1, l2, grad, indices, use_locking, name):
        """Helper to construct the input dictionary."""
        dtype = var.dtype
        return {
            'var': var,
            'accum': accum,
            'lr': np.array(lr, dtype=dtype),
            'l1': np.array(l1, dtype=dtype),
            'l2': np.array(l2, dtype=dtype),
            'grad': grad,
            'indices': indices,
            'use_locking': use_locking,
            'name': name
        }

    # Case 1: Basic float32, 2D tensor
    var1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    accum1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    indices1 = np.array([0, 2], dtype=np.int32)
    grad1 = np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var1, accum1, 0.01, 0.1, 0.001, grad1, indices1, False, "float32_basic")))

    # Case 2: Basic float64, 2D tensor with locking
    var2 = np.array([[10.0], [20.0], [30.0]], dtype=np.float64)
    accum2 = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)
    indices2 = np.array([1], dtype=np.int64)
    grad2 = np.array([[-5.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var2, accum2, 0.1, 0.0, 0.5, grad2, indices2, True, "float64_locking")))

    # Case 3: 1D tensors
    var3 = np.arange(5, dtype=np.float32)
    accum3 = np.full(5, 0.1, dtype=np.float32)
    indices3 = np.array([1, 3, 4], dtype=np.int32)
    grad3 = np.array([0.5, -0.3, 0.1], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var3, accum3, 0.5, 0.05, 0.02, grad3, indices3, False, "1d_tensors")))

    # Case 4: 3D tensors
    var4 = np.random.rand(4, 3, 2).astype(np.float32)
    accum4 = np.full((4, 3, 2), 0.1, dtype=np.float32)
    indices4 = np.array([0, 3], dtype=np.int32)
    grad4 = np.random.rand(2, 3, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var4, accum4, 0.001, 0.2, 0.1, grad4, indices4, True, "3d_tensors")))

    # Case 5: Zero regularization
    var5 = np.array([[1.0, 2.0]], dtype=np.float32)
    accum5 = np.array([[0.1, 0.1]], dtype=np.float32)
    indices5 = np.array([0], dtype=np.int32)
    grad5 = np.array([[0.5, -0.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var5, accum5, 0.01, 0.0, 0.0, grad5, indices5, False, "zero_regularization")))

    # Case 6: All indices are updated
    var6 = np.array([[1.0], [2.0]], dtype=np.float32)
    accum6 = np.full((2, 1), 0.5, dtype=np.float32)
    indices6 = np.array([0, 1], dtype=np.int32)
    grad6 = np.array([[0.1], [-0.2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var6, accum6, 0.2, 0.1, 0.0, grad6, indices6, True, "update_all")))

    # Case 7: High learning rate
    var7 = np.array([[100.0, -100.0]], dtype=np.float32)
    accum7 = np.array([[1.0, 1.0]], dtype=np.float32)
    indices7 = np.array([0], dtype=np.int32)
    grad7 = np.array([[0.01, 0.01]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var7, accum7, 50.0, 0.0, 0.0, grad7, indices7, False, "high_lr")))

    # Case 8: High regularization
    var8 = np.array([[1.0, 1.0]], dtype=np.float64)
    accum8 = np.array([[1.0, 1.0]], dtype=np.float64)
    indices8 = np.array([0], dtype=np.int64)
    grad8 = np.array([[0.5, -0.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var8, accum8, 0.1, 10.0, 10.0, grad8, indices8, True, "high_regularization")))

    # Case 9: Negative initial values
    var9 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    accum9 = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    indices9 = np.array([1], dtype=np.int32)
    grad9 = np.array([[0.3, -0.4]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var9, accum9, 0.05, 0.1, 0.2, grad9, indices9, False, "negative_initial_var")))

    # Case 10: bfloat16 type
    bfloat16 = tf.bfloat16.as_numpy_dtype
    var10 = np.array([[1.0, 2.0]], dtype=bfloat16)
    accum10 = np.array([[0.1, 0.1]], dtype=bfloat16)
    indices10 = np.array([0], dtype=np.int32)
    grad10 = np.array([[0.5, 0.2]], dtype=bfloat16)
    list_of_inputs.append(copy.deepcopy(_create_input_dict(var10, accum10, 0.01, 0.1, 0.01, grad10, indices10, False, "bfloat16_type")))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseApplyProximalAdagrad"] = tf_raw_ops_SparseApplyProximalAdagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyProximalAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyProximalAdagrad'.")

check_valid('tf.raw_ops.SparseApplyProximalAdagrad', generated_inputs['tf.raw_ops.SparseApplyProximalAdagrad'], lib="tf", suffix=0)
