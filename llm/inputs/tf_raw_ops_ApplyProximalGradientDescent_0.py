
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_proximal_gradient_descent_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.01, dtype=np.float32)
    l2 = np.array(0.02, dtype=np.float32)
    delta = np.array([0.5, -0.2, 0.1], dtype=np.float32)
    use_locking = False
    name = "test_op_1"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    alpha = np.array(0.05, dtype=np.float64)
    l1 = np.array(0.005, dtype=np.float64)
    l2 = np.array(0.01, dtype=np.float64)
    delta = np.array([[0.2, -0.1], [0.3, 0.05]], dtype=np.float64)
    use_locking = True
    name = "test_op_2"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 3
    var = np.array([1, 2, 3], dtype=np.int32)
    alpha = np.array(0.2, dtype=np.int32)
    l1 = np.array(1, dtype=np.int32)
    l2 = np.array(2, dtype=np.int32)
    delta = np.array([-1, 0, 1], dtype=np.int32)
    use_locking = False
    name = "test_op_3"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 4
    var = np.array([1, 2, 3], dtype=np.int64)
    alpha = np.array(1, dtype=np.int64)
    l1 = np.array(1, dtype=np.int64)
    l2 = np.array(1, dtype=np.int64)
    delta = np.array([1, 2, 3], dtype=np.int64)
    use_locking = True
    name = "test_op_4"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 5
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.01, dtype=np.float32)
    l2 = np.array(0.02, dtype=np.float32)
    delta = np.array([0.5, -0.2, 0.1], dtype=np.float32)
    use_locking = False
    name = "test_op_5"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 6
    var = np.array([1, 2, 3], dtype=np.int32)
    alpha = np.array(0, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    delta = np.array([0, 0, 0], dtype=np.int32)
    use_locking = True
    name = "test_op_6"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 7
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.01, dtype=np.float32)
    l2 = np.array(0.02, dtype=np.float32)
    delta = np.array([[0.5, -0.2], [0.1, 0.3]], dtype=np.float32)
    use_locking = False
    name = "test_op_7"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 8
    var = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    alpha = np.array(1, dtype=np.int32)
    l1 = np.array(1, dtype=np.int32)
    l2 = np.array(1, dtype=np.int32)
    delta = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    use_locking = True
    name = "test_op_8"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 9
    var = np.array([1.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.01, dtype=np.float32)
    l2 = np.array(0.02, dtype=np.float32)
    delta = np.array([0.5], dtype=np.float32)
    use_locking = False
    name = "test_op_9"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 10
    var = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    alpha = np.array(2, dtype=np.int32)
    l1 = np.array(1, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    delta = np.array([0, 1, -1, 2, -2], dtype=np.int32)
    use_locking = True
    name = "test_op_10"
    input_dict = {"var": var, "alpha": alpha, "l1": l1, "l2": l2, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    abstract_list = []
    for i in list_of_inputs:
        abstract_list.append({"args": [], "kwargs": i})

    return abstract_list

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyProximalGradientDescent"] = tf_raw_ops_apply_proximal_gradient_descent_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyProximalGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyProximalGradientDescent'.")

check_valid('tf.raw_ops.ApplyProximalGradientDescent', generated_inputs['tf.raw_ops.ApplyProximalGradientDescent'], lib="tf", suffix=0)
