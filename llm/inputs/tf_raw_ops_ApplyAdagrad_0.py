
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def get_tf_raw_ops_apply_adagrad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ApplyAdagrad.
    To fix the "does not support eager execution" error, the `var` and `accum`
    inputs must be tf.Variable objects, as the operation modifies them in-place.
    """
    input_list = []

    # Test case 1: Basic float32, 1D
    var_1 = tf.Variable([1.0, 2.0], dtype=tf.float32)
    accum_1 = tf.Variable([0.1, 0.1], dtype=tf.float32)
    lr_1 = tf.constant(0.001, dtype=tf.float32)
    grad_1 = tf.constant([0.2, -0.3], dtype=tf.float32)
    input_list.append({
        'var': var_1,
        'accum': accum_1,
        'lr': lr_1,
        'grad': grad_1,
        'use_locking': True,
        'update_slots': True,
        'name': 'test_float32'
    })

    # Test case 2: float64, 2D
    var_2 = tf.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64)
    accum_2 = tf.Variable([[0.2, 0.3], [0.4, 0.5]], dtype=tf.float64)
    lr_2 = tf.constant(0.01, dtype=tf.float64)
    grad_2 = tf.constant([[-0.1, 0.2], [0.3, -0.4]], dtype=tf.float64)
    input_list.append({
        'var': var_2,
        'accum': accum_2,
        'lr': lr_2,
        'grad': grad_2,
        'use_locking': False,
        'update_slots': True,
        'name': 'test_float64'
    })

    # Test case 3: half (float16), 3D
    var_3 = tf.Variable(np.random.randn(2, 2, 2).astype(np.float16))
    accum_3 = tf.Variable(np.abs(np.random.randn(2, 2, 2).astype(np.float16)) + 0.1)
    lr_3 = tf.constant(0.1, dtype=tf.float16)
    grad_3 = tf.constant(np.random.randn(2, 2, 2).astype(np.float16))
    input_list.append({
        'var': var_3,
        'accum': accum_3,
        'lr': lr_3,
        'grad': grad_3,
        'use_locking': True,
        'update_slots': False,
        'name': 'test_float16_no_update'
    })

    # Test case 4: complex64
    var_4 = tf.Variable([1+2j, 3-1j], dtype=tf.complex64)
    accum_4 = tf.Variable([0.1+0j, 0.2+0j], dtype=tf.complex64)
    lr_4 = tf.constant(0.1+0j, dtype=tf.complex64)
    grad_4 = tf.constant([0.5+0.5j, -0.2-0.3j], dtype=tf.complex64)
    input_list.append({
        'var': var_4,
        'accum': accum_4,
        'lr': lr_4,
        'grad': grad_4,
        'use_locking': False,
        'update_slots': True,
        'name': 'test_complex64'
    })

    # Test case 5: complex128
    var_5 = tf.Variable([[1+2j]], dtype=tf.complex128)
    accum_5 = tf.Variable([[0.5+0j]], dtype=tf.complex128)
    lr_5 = tf.constant(0.05+0j, dtype=tf.complex128)
    grad_5 = tf.constant([[0.1+0.2j]], dtype=tf.complex128)
    input_list.append({
        'var': var_5,
        'accum': accum_5,
        'lr': lr_5,
        'grad': grad_5,
        'use_locking': True,
        'update_slots': True,
        'name': 'test_complex128'
    })

    # Test case 6: bfloat16
    bf16_dtype = tf.bfloat16
    var_6 = tf.Variable(np.array([1.0, 2.0]), dtype=bf16_dtype)
    accum_6 = tf.Variable(np.array([0.1, 0.1]), dtype=bf16_dtype)
    lr_6 = tf.constant(0.01, dtype=bf16_dtype)
    grad_6 = tf.constant([0.5, -0.5], dtype=bf16_dtype)
    input_list.append({
        'var': var_6,
        'accum': accum_6,
        'lr': lr_6,
        'grad': grad_6,
        'use_locking': False,
        'update_slots': False,
        'name': 'test_bfloat16'
    })

    # Test case 7: Scalar (0-D tensor) float32
    var_7 = tf.Variable(10.0, dtype=tf.float32)
    accum_7 = tf.Variable(1.0, dtype=tf.float32)
    lr_7 = tf.constant(0.1, dtype=tf.float32)
    grad_7 = tf.constant(-2.0, dtype=tf.float32)
    input_list.append({
        'var': var_7,
        'accum': accum_7,
        'lr': lr_7,
        'grad': grad_7,
        'use_locking': True,
        'update_slots': True,
        'name': 'test_scalar_float32'
    })

    # Test case 8: Larger tensors, float32
    var_8 = tf.Variable(np.random.randn(5, 5).astype(np.float32))
    accum_8 = tf.Variable(np.abs(np.random.randn(5, 5).astype(np.float32)) + 0.1)
    lr_8 = tf.constant(0.001, dtype=tf.float32)
    grad_8 = tf.constant(np.random.randn(5, 5).astype(np.float32))
    input_list.append({
        'var': var_8,
        'accum': accum_8,
        'lr': lr_8,
        'grad': grad_8,
        'use_locking': False,
        'update_slots': True,
        'name': 'test_large_tensors'
    })

    # Test case 9: Zero gradient
    var_9 = tf.Variable([1.0, -1.0], dtype=tf.float32)
    accum_9 = tf.Variable([1.0, 1.0], dtype=tf.float32)
    lr_9 = tf.constant(0.1, dtype=tf.float32)
    grad_9 = tf.constant([0.0, 0.0], dtype=tf.float32)
    input_list.append({
        'var': var_9,
        'accum': accum_9,
        'lr': lr_9,
        'grad': grad_9,
        'use_locking': True,
        'update_slots': True,
        'name': 'test_zero_grad'
    })

    # Test case 10: High learning rate
    var_10 = tf.Variable([[0.5], [0.5]], dtype=tf.float32)
    accum_10 = tf.Variable([[0.01], [0.01]], dtype=tf.float32)
    lr_10 = tf.constant(10.0, dtype=tf.float32)
    grad_10 = tf.constant([[0.1], [-0.1]], dtype=tf.float32)
    input_list.append({
        'var': var_10,
        'accum': accum_10,
        'lr': lr_10,
        'grad': grad_10,
        'use_locking': False,
        'update_slots': True,
        'name': 'test_high_lr'
    })

    return input_list

generated_inputs["tf.raw_ops.ApplyAdagrad"] = get_tf_raw_ops_apply_adagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagrad'.")

check_valid('tf.raw_ops.ApplyAdagrad', generated_inputs['tf.raw_ops.ApplyAdagrad'], lib="tf", suffix=0)
