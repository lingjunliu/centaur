
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_apply_ftrl_inputs():
    list_of_inputs = []

    # Input 1
    var = tf.compat.v1.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), use_resource=True)
    accum = tf.compat.v1.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32), use_resource=True)
    linear = tf.compat.v1.Variable(np.array([0.4, 0.5, 0.6], dtype=np.float32), use_resource=True)
    grad = tf.constant(np.array([0.7, 0.8, 0.9], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    l1 = tf.constant(np.array(0.0, dtype=np.float32))
    l2 = tf.constant(np.array(0.0, dtype=np.float32))
    lr_power = tf.constant(np.array(-0.5, dtype=np.float32))
    use_locking = False
    multiply_linear_by_lr = False
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = tf.compat.v1.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), use_resource=True)
    accum = tf.compat.v1.Variable(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64), use_resource=True)
    linear = tf.compat.v1.Variable(np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float64), use_resource=True)
    grad = tf.constant(np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float64))
    lr = tf.constant(np.array(0.001, dtype=np.float64))
    l1 = tf.constant(np.array(0.1, dtype=np.float64))
    l2 = tf.constant(np.array(0.01, dtype=np.float64))
    lr_power = tf.constant(np.array(-0.1, dtype=np.float64))
    use_locking = True
    multiply_linear_by_lr = True
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = tf.compat.v1.Variable(np.array([1, 2, 3], dtype=np.int32), use_resource=True)
    accum = tf.compat.v1.Variable(np.array([0, 1, 2], dtype=np.int32), use_resource=True)
    linear = tf.compat.v1.Variable(np.array([3, 4, 5], dtype=np.int32), use_resource=True)
    grad = tf.constant(np.array([6, 7, 8], dtype=np.int32))
    lr = tf.constant(np.array(1, dtype=np.int32))
    l1 = tf.constant(np.array(0, dtype=np.int32))
    l2 = tf.constant(np.array(0, dtype=np.int32))
    lr_power = tf.constant(np.array(0, dtype=np.int32))
    use_locking = False
    multiply_linear_by_lr = False
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = tf.compat.v1.Variable(np.array([-1.0, -2.0], dtype=np.float32), use_resource=True)
    accum = tf.compat.v1.Variable(np.array([0.5, 0.6], dtype=np.float32), use_resource=True)
    linear = tf.compat.v1.Variable(np.array([-0.7, -0.8], dtype=np.float32), use_resource=True)
    grad = tf.constant(np.array([-0.9, -1.0], dtype=np.float32))
    lr = tf.constant(np.array(0.02, dtype=np.float32))
    l1 = tf.constant(np.array(0.01, dtype=np.float32))
    l2 = tf.constant(np.array(0.001, dtype=np.float32))
    lr_power = tf.constant(np.array(-0.2, dtype=np.float32))
    use_locking = True
    multiply_linear_by_lr = False
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = tf.compat.v1.Variable(np.array([1.0], dtype=np.float32), use_resource=True)
    accum = tf.compat.v1.Variable(np.array([0.1], dtype=np.float32), use_resource=True)
    linear = tf.compat.v1.Variable(np.array([0.4], dtype=np.float32), use_resource=True)
    grad = tf.constant(np.array([0.7], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    l1 = tf.constant(np.array(0.0, dtype=np.float32))
    l2 = tf.constant(np.array(0.0, dtype=np.float32))
    lr_power = tf.constant(np.array(0.0, dtype=np.float32))
    use_locking = False
    multiply_linear_by_lr = True
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - different shape
    var = tf.compat.v1.Variable(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    accum = tf.compat.v1.Variable(np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32))
    linear = tf.compat.v1.Variable(np.array([[[0.4, 0.5], [0.6, 0.7]], [[0.8, 0.9], [1.0, 1.1]]], dtype=np.float32))
    grad = tf.constant(np.array([[[0.7, 0.8], [0.9, 1.0]], [[1.1, 1.2], [1.3, 1.4]]], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    l1 = tf.constant(np.array(0.0, dtype=np.float32))
    l2 = tf.constant(np.array(0.0, dtype=np.float32))
    lr_power = tf.constant(np.array(-0.5, dtype=np.float32))
    use_locking = False
    multiply_linear_by_lr = False
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = tf.compat.v1.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    accum = tf.compat.v1.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float64))
    linear = tf.compat.v1.Variable(np.array([0.4, 0.5, 0.6], dtype=np.float64))
    grad = tf.constant(np.array([0.7, 0.8, 0.9], dtype=np.float64))
    lr = tf.constant(np.array(0.01, dtype=np.float64))
    l1 = tf.constant(np.array(0.0, dtype=np.float64))
    l2 = tf.constant(np.array(0.0, dtype=np.float64))
    lr_power = tf.constant(np.array(-0.5, dtype=np.float64))
    use_locking = False
    multiply_linear_by_lr = False
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: integer values
    var = tf.compat.v1.Variable(np.array([1, 2], dtype=np.int32))
    accum = tf.compat.v1.Variable(np.array([0, 0], dtype=np.int32))
    linear = tf.compat.v1.Variable(np.array([0, 0], dtype=np.int32))
    grad = tf.constant(np.array([1, 1], dtype=np.int32))
    lr = tf.constant(np.array(1, dtype=np.int32))
    l1 = tf.constant(np.array(0, dtype=np.int32))
    l2 = tf.constant(np.array(0, dtype=np.int32))
    lr_power = tf.constant(np.array(0, dtype=np.int32))
    use_locking = False
    multiply_linear_by_lr = False
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: l1 regularization
    var = tf.compat.v1.Variable(np.array([1.0, 2.0], dtype=np.float32))
    accum = tf.compat.v1.Variable(np.array([0.1, 0.2], dtype=np.float32))
    linear = tf.compat.v1.Variable(np.array([0.4, 0.5], dtype=np.float32))
    grad = tf.constant(np.array([0.7, 0.8], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    l1 = tf.constant(np.array(0.1, dtype=np.float32))
    l2 = tf.constant(np.array(0.0, dtype=np.float32))
    lr_power = tf.constant(np.array(-0.5, dtype=np.float32))
    use_locking = False
    multiply_linear_by_lr = False
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: l2 regularization
    var = tf.compat.v1.Variable(np.array([1.0, 2.0], dtype=np.float32))
    accum = tf.compat.v1.Variable(np.array([0.1, 0.2], dtype=np.float32))
    linear = tf.compat.v1.Variable(np.array([0.4, 0.5], dtype=np.float32))
    grad = tf.constant(np.array([0.7, 0.8], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    l1 = tf.constant(np.array(0.0, dtype=np.float32))
    l2 = tf.constant(np.array(0.1, dtype=np.float32))
    lr_power = tf.constant(np.array(-0.5, dtype=np.float32))
    use_locking = False
    multiply_linear_by_lr = False
    name = None

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyFtrl"] = tf_raw_ops_apply_ftrl_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyFtrl' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyFtrl'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyFtrl', generated_inputs['tf.raw_ops.ApplyFtrl'], lib="tf", suffix=0)
