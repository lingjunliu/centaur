
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_apply_gradient_descent_inputs():
    list_of_inputs = []

    # Input 1
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    alpha = tf.constant(0.1, dtype=np.float32)
    delta = tf.constant([0.5, 0.5, 0.5], dtype=np.float32)
    use_locking = False
    name = "gd_1"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    alpha = tf.constant(0.01, dtype=np.float64)
    delta = tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    use_locking = True
    name = "gd_2"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    alpha = tf.constant(1, dtype=np.int32)
    delta = tf.constant([1, 1, 1], dtype=np.int32)
    use_locking = False
    name = "gd_3"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = tf.Variable(np.array([255], dtype=np.uint8))
    alpha = tf.constant(10, dtype=np.uint8)
    delta = tf.constant([5], dtype=np.uint8)
    use_locking = True
    name = "gd_4"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = tf.Variable(np.array([-32768], dtype=np.int16))
    alpha = tf.constant(100, dtype=np.int16)
    delta = tf.constant([10], dtype=np.int16)
    use_locking = False
    name = "gd_5"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = tf.Variable(np.array([-128], dtype=np.int8))
    alpha = tf.constant(1, dtype=np.int8)
    delta = tf.constant([1], dtype=np.int8)
    use_locking = True
    name = "gd_6"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = tf.Variable(np.array([1+1j, 2+2j], dtype=np.complex64))
    alpha = tf.constant(0.1+0j, dtype=np.complex64)
    delta = tf.constant([0.5+0j, 0.5+0j], dtype=np.complex64)
    use_locking = False
    name = "gd_7"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int64))
    alpha = tf.constant(1, dtype=np.int64)
    delta = tf.constant([1, 1, 1], dtype=np.int64)
    use_locking = True
    name = "gd_8"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    alpha = tf.constant(0.1, dtype=np.float16)
    delta = tf.constant([0.5, 0.5, 0.5], dtype=np.float16)
    use_locking = False
    name = "gd_9"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "delta": delta,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = tf.Variable(np.array([1+1j, 2+2j], dtype=np.complex128))
    alpha = tf.constant(0.1+0j, dtype=np.complex128)
    delta = tf.constant([0.5+0j, 0.5+0j], dtype=np.complex128)
    use_locking = True
    name = "gd_10"

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
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyGradientDescent'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyGradientDescent', generated_inputs['tf.raw_ops.ApplyGradientDescent'], lib="tf", suffix=0)
