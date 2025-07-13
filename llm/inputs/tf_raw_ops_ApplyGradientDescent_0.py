
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_gradient_descent_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    alpha = tf.constant(0.1, dtype=np.float32)
    delta = tf.constant([0.5, 1.0, 1.5], dtype=np.float32)
    use_locking = False
    name = "grad_descent_1"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different shape
    var = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    alpha = tf.constant(0.05, dtype=np.float64)
    delta = tf.constant([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    use_locking = True
    name = "grad_descent_2"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    alpha = tf.constant(1, dtype=np.int32)
    delta = tf.constant([1, 1, 1], dtype=np.int32)
    use_locking = False
    name = "grad_descent_3"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative alpha and delta
    var = tf.Variable(np.array([5.0, 6.0], dtype=np.float32))
    alpha = tf.constant(-0.2, dtype=np.float32)
    delta = tf.constant([-1.0, -2.0], dtype=np.float32)
    use_locking = True
    name = "grad_descent_4"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: alpha close to 0
    var = tf.Variable(np.array([7.0, 8.0], dtype=np.float32))
    alpha = tf.constant(0.001, dtype=np.float32)
    delta = tf.constant([1.0, 2.0], dtype=np.float32)
    use_locking = False
    name = "grad_descent_5"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: int64, different values
    var = tf.Variable(np.array([5, 10, 15], dtype=np.int64))
    alpha = tf.constant(2, dtype=np.int64)
    delta = tf.constant([1, 2, 3], dtype=np.int64)
    use_locking = True
    name = "grad_descent_6"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64
    var = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    alpha = tf.constant(0.5+0.5j, dtype=np.complex64)
    delta = tf.constant([0.5+0.5j, 1+1j, 1.5+1.5j], dtype=np.complex64)
    use_locking = False
    name = "grad_descent_7"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    alpha = tf.constant(0.1, dtype=np.float16)
    delta = tf.constant([0.5, 1.0, 1.5], dtype=np.float16)
    use_locking = True
    name = "grad_descent_8"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    # Input 9: uint8
    var = tf.Variable(np.array([1, 2, 3], dtype=np.uint8))
    alpha = tf.constant(1, dtype=np.uint8)
    delta = tf.constant([1, 1, 1], dtype=np.uint8)
    use_locking = False
    name = "grad_descent_9"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128
    var = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128))
    alpha = tf.constant(0.5+0.5j, dtype=np.complex128)
    delta = tf.constant([0.5+0.5j, 1+1j, 1.5+1.5j], dtype=np.complex128)
    use_locking = False
    name = "grad_descent_10"
    input_dict = {"var": np.array(var), "alpha": alpha, "delta": delta, "use_locking": use_locking, "name": name}
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
