
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adadelta_inputs():
    list_of_inputs = []

    # Input 1
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    accum_update = tf.Variable(np.array([0.01, 0.02, 0.03], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    rho = tf.constant(np.array(0.95, dtype=np.float32))
    epsilon = tf.constant(np.array(1e-06, dtype=np.float32))
    grad = tf.constant(np.array([0.5, 0.6, 0.7], dtype=np.float32))
    use_locking = False
    name = "adadelta_1"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    accum = tf.Variable(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))
    accum_update = tf.Variable(np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64))
    lr = tf.constant(np.array(0.01, dtype=np.float64))
    rho = tf.constant(np.array(0.95, dtype=np.float64))
    epsilon = tf.constant(np.array(1e-06, dtype=np.float64))
    grad = tf.constant(np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64))
    use_locking = True
    name = "adadelta_2"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    accum = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    accum_update = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    lr = tf.constant(np.array(1, dtype=np.int32))
    rho = tf.constant(np.array(1, dtype=np.int32))
    epsilon = tf.constant(np.array(1, dtype=np.int32))
    grad = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    use_locking = False
    name = "adadelta_3"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = tf.Variable(np.array([-1.0, -2.0, -3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    accum_update = tf.Variable(np.array([0.01, 0.02, 0.03], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    rho = tf.constant(np.array(0.95, dtype=np.float32))
    epsilon = tf.constant(np.array(1e-06, dtype=np.float32))
    grad = tf.constant(np.array([0.5, 0.6, 0.7], dtype=np.float32))
    use_locking = False
    name = "adadelta_4"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    accum_update = tf.Variable(np.array([0.01, 0.02, 0.03], dtype=np.float32))
    lr = tf.constant(np.array(-0.01, dtype=np.float32))
    rho = tf.constant(np.array(0.95, dtype=np.float32))
    epsilon = tf.constant(np.array(1e-06, dtype=np.float32))
    grad = tf.constant(np.array([0.5, 0.6, 0.7], dtype=np.float32))
    use_locking = False
    name = "adadelta_5"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    accum_update = tf.Variable(np.array([0.01, 0.02, 0.03], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    rho = tf.constant(np.array(-0.95, dtype=np.float32))
    epsilon = tf.constant(np.array(1e-06, dtype=np.float32))
    grad = tf.constant(np.array([0.5, 0.6, 0.7], dtype=np.float32))
    use_locking = False
    name = "adadelta_6"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    accum_update = tf.Variable(np.array([0.01, 0.02, 0.03], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    rho = tf.constant(np.array(0.95, dtype=np.float32))
    epsilon = tf.constant(np.array(-1e-06, dtype=np.float32))
    grad = tf.constant(np.array([0.5, 0.6, 0.7], dtype=np.float32))
    use_locking = False
    name = "adadelta_7"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    accum_update = tf.Variable(np.array([0.01, 0.02, 0.03], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    rho = tf.constant(np.array(0.95, dtype=np.float32))
    epsilon = tf.constant(np.array(1e-06, dtype=np.float32))
    grad = tf.constant(np.array([-0.5, -0.6, -0.7], dtype=np.float32))
    use_locking = False
    name = "adadelta_8"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int32))
    accum = tf.Variable(np.array([[5, 6], [7, 8]], dtype=np.int32))
    accum_update = tf.Variable(np.array([[9, 10], [11, 12]], dtype=np.int32))
    lr = tf.constant(np.array(2, dtype=np.int32))
    rho = tf.constant(np.array(3, dtype=np.int32))
    epsilon = tf.constant(np.array(4, dtype=np.int32))
    grad = tf.constant(np.array([[13, 14], [15, 16]], dtype=np.int32))
    use_locking = True
    name = "adadelta_9"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = tf.Variable(np.array([1.0], dtype=np.float32))
    accum = tf.Variable(np.array([0.1], dtype=np.float32))
    accum_update = tf.Variable(np.array([0.01], dtype=np.float32))
    lr = tf.constant(np.array(0.01, dtype=np.float32))
    rho = tf.constant(np.array(0.95, dtype=np.float32))
    epsilon = tf.constant(np.array(1e-06, dtype=np.float32))
    grad = tf.constant(np.array([0.5], dtype=np.float32))
    use_locking = False
    name = "adadelta_10"

    input_dict = {
        "var": var.numpy(),
        "accum": accum.numpy(),
        "accum_update": accum_update.numpy(),
        "lr": lr.numpy(),
        "rho": rho.numpy(),
        "epsilon": epsilon.numpy(),
        "grad": grad.numpy(),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyAdadelta"] = tf_raw_ops_apply_adadelta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdadelta'.")

check_valid('tf.raw_ops.ApplyAdadelta', generated_inputs['tf.raw_ops.ApplyAdadelta'], lib="tf", suffix=0)
