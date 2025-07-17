
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_proximal_gradient_descent_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 example
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    delta = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with regularization
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    alpha = np.array(0.1, dtype=np.float64)
    l1 = np.array(0.1, dtype=np.float64)
    l2 = np.array(0.1, dtype=np.float64)
    delta = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    use_locking = True
    name = "prox_grad_descent_2"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32 example
    var = np.array([1, 2, 3], dtype=np.int32)
    alpha = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    delta = np.array([1, 1, 1], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative delta values
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    delta = np.array([-0.5, -1.0, -1.5], dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5: Higher alpha and l1, l2 values
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha = np.array(0.5, dtype=np.float32)
    l1 = np.array(0.2, dtype=np.float32)
    l2 = np.array(0.2, dtype=np.float32)
    delta = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D Tensor
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    delta = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8 example
    var = np.array([1, 2, 3], dtype=np.uint8)
    alpha = np.array(1, dtype=np.uint8)
    l1 = np.array(0, dtype=np.uint8)
    l2 = np.array(0, dtype=np.uint8)
    delta = np.array([1, 1, 1], dtype=np.uint8)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small values
    var = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    alpha = np.array(0.0001, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    delta = np.array([0.0005, 0.001, 0.0015], dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D Tensor with l1 and l2
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.01, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    delta = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64 example
    var = np.array([1, 2, 3], dtype=np.int64)
    alpha = np.array(1, dtype=np.int64)
    l1 = np.array(0, dtype=np.int64)
    l2 = np.array(0, dtype=np.int64)
    delta = np.array([1, 1, 1], dtype=np.int64)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

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
