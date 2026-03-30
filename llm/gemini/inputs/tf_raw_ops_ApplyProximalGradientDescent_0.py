
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_apply_proximal_gradient_descent_inputs():
    list_of_inputs = []

    # Input 1: float32, basic test
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    alpha = tf.constant(0.1, dtype=np.float32)
    l1 = tf.constant(0.01, dtype=np.float32)
    l2 = tf.constant(0.02, dtype=np.float32)
    delta = tf.constant([0.5, 0.5, 0.5], dtype=np.float32)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": False, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values
    var = tf.Variable(np.array([-1.0, 2.5, -3.2], dtype=np.float64))
    alpha = tf.constant(0.2, dtype=np.float64)
    l1 = tf.constant(0.05, dtype=np.float64)
    l2 = tf.constant(0.1, dtype=np.float64)
    delta = tf.constant([0.1, -0.2, 0.3], dtype=np.float64)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": True, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, positive values
    var = tf.Variable(np.array([10, 20, 30], dtype=np.int32))
    alpha = tf.constant(1, dtype=np.int32)
    l1 = tf.constant(2, dtype=np.int32)
    l2 = tf.constant(1, dtype=np.int32)
    delta = tf.constant([5, 5, 5], dtype=np.int32)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": False, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, negative values
    var = tf.Variable(np.array([-10, 20, -30], dtype=np.int64))
    alpha = tf.constant(2, dtype=np.int64)
    l1 = tf.constant(1, dtype=np.int64)
    l2 = tf.constant(3, dtype=np.int64)
    delta = tf.constant([-5, 5, -5], dtype=np.int64)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": True, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float16
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    alpha = tf.constant(0.1, dtype=np.float16)
    l1 = tf.constant(0.01, dtype=np.float16)
    l2 = tf.constant(0.02, dtype=np.float16)
    delta = tf.constant([0.5, 0.5, 0.5], dtype=np.float16)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": False, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint32
    var = tf.Variable(np.array([10, 20, 30], dtype=np.uint32))
    alpha = tf.constant(1, dtype=np.uint32)
    l1 = tf.constant(2, dtype=np.uint32)
    l2 = tf.constant(1, dtype=np.uint32)
    delta = tf.constant([5, 5, 5], dtype=np.uint32)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": False, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint64
    var = tf.Variable(np.array([10, 20, 30], dtype=np.uint64))
    alpha = tf.constant(1, dtype=np.uint64)
    l1 = tf.constant(2, dtype=np.uint64)
    l2 = tf.constant(1, dtype=np.uint64)
    delta = tf.constant([5, 5, 5], dtype=np.uint64)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": False, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, shape (2,3)
    var = tf.Variable(np.array([[1.0, 2.0, 3.0],[4.0,5.0,6.0]], dtype=np.float32))
    alpha = tf.constant(0.1, dtype=np.float32)
    l1 = tf.constant(0.01, dtype=np.float32)
    l2 = tf.constant(0.02, dtype=np.float32)
    delta = tf.constant([[0.5, 0.5, 0.5],[0.1,0.2,0.3]], dtype=np.float32)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": False, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, shape (3,1)
    var = tf.Variable(np.array([[-1.0], [2.5], [-3.2]], dtype=np.float64))
    alpha = tf.constant(0.2, dtype=np.float64)
    l1 = tf.constant(0.05, dtype=np.float64)
    l2 = tf.constant(0.1, dtype=np.float64)
    delta = tf.constant([[0.1], [-0.2], [0.3]], dtype=np.float64)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": True, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32, shape (1,1)
    var = tf.Variable(np.array([[10]], dtype=np.int32))
    alpha = tf.constant(1, dtype=np.int32)
    l1 = tf.constant(2, dtype=np.int32)
    l2 = tf.constant(1, dtype=np.int32)
    delta = tf.constant([[5]], dtype=np.int32)
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": False, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyProximalGradientDescent', generated_inputs['tf.raw_ops.ApplyProximalGradientDescent'], lib="tf", suffix=0)
