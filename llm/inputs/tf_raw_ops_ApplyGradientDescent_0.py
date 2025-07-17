
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_gradient_descent_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    alpha = np.array(0.1, dtype=np.float32)
    delta = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    use_locking = False
    name = "gradient_descent_1"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different alpha, float64
    var = tf.Variable(np.array([4.0, 5.0, 6.0], dtype=np.float64))
    alpha = np.array(0.01, dtype=np.float64)
    delta = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    use_locking = True
    name = "gradient_descent_2"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32
    var = tf.Variable(np.array([10, 20, 30], dtype=np.int32))
    alpha = np.array(2, dtype=np.int32)
    delta = np.array([1, 2, 3], dtype=np.int32)
    use_locking = False
    name = "gradient_descent_3"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative delta, float32
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    alpha = np.array(0.5, dtype=np.float32)
    delta = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    use_locking = True
    name = "gradient_descent_4"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional, float32
    var = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    alpha = np.array(0.2, dtype=np.float32)
    delta = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    use_locking = False
    name = "gradient_descent_5"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64
    var = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    alpha = np.array(0.1+0j, dtype=np.complex64)
    delta = np.array([0.5+0.5j, 0.5+0.5j, 0.5+0.5j], dtype=np.complex64)
    use_locking = True
    name = "gradient_descent_6"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    alpha = np.array(0.1, dtype=np.float16)
    delta = np.array([0.5, 0.5, 0.5], dtype=np.float16)
    use_locking = True
    name = "gradient_descent_8"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8
    var = tf.Variable(np.array([10, 20, 30], dtype=np.uint8))
    alpha = np.array(1, dtype=np.uint8)
    delta = np.array([1, 2, 3], dtype=np.uint8)
    use_locking = False
    name = "gradient_descent_9"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint32
    var = tf.Variable(np.array([10, 20, 30], dtype=np.uint32))
    alpha = np.array(1, dtype=np.uint32)
    delta = np.array([1, 2, 3], dtype=np.uint32)
    use_locking = True
    name = "gradient_descent_10"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyGradientDescent"] = tf_raw_ops_apply_gradient_descent_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyGradientDescent'.")

check_valid('tf.raw_ops.ApplyGradientDescent', generated_inputs['tf.raw_ops.ApplyGradientDescent'], lib="tf", suffix=0)
