
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_apply_adadelta_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0], dtype=np.float32)
    accum = np.array([0.1, 0.2], dtype=np.float32)
    accum_update = np.array([0.3, 0.4], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    use_locking = False
    name = "adadelta_1"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "accum": tf.Variable(accum, dtype=tf.float32),
        "accum_update": tf.Variable(accum_update, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    accum_update = np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    epsilon = np.array(1e-6, dtype=np.float64)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    use_locking = True
    name = "adadelta_2"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float64),
        "accum": tf.Variable(accum, dtype=tf.float64),
        "accum_update": tf.Variable(accum_update, dtype=tf.float64),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float64),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float64),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float64),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([4, 5, 6], dtype=np.int32)
    accum_update = np.array([7, 8, 9], dtype=np.int32)
    lr = np.array(0.1, dtype=np.int32)
    rho = np.array(0.8, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([10, 11, 12], dtype=np.int32)
    use_locking = False
    name = "adadelta_3"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int32),
        "accum": tf.Variable(accum, dtype=tf.int32),
        "accum_update": tf.Variable(accum_update, dtype=tf.int32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.int32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.int32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([1.0, 2.0], dtype=np.float32)
    accum = np.array([0.1, 0.2], dtype=np.float32)
    accum_update = np.array([0.3, 0.4], dtype=np.float32)
    lr = np.array(-0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    use_locking = False
    name = "adadelta_4"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "accum": tf.Variable(accum, dtype=tf.float32),
        "accum_update": tf.Variable(accum_update, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    var = np.array(1.0, dtype=np.float32)
    accum = np.array(0.1, dtype=np.float32)
    accum_update = np.array(0.3, dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array(0.5, dtype=np.float32)
    use_locking = False
    name = "adadelta_5"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "accum": tf.Variable(accum, dtype=tf.float32),
        "accum_update": tf.Variable(accum_update, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([1, 2, 3], dtype=np.int64)
    accum = np.array([4, 5, 6], dtype=np.int64)
    accum_update = np.array([7, 8, 9], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    rho = np.array(1, dtype=np.int64)
    epsilon = np.array(1, dtype=np.int64)
    grad = np.array([10, 11, 12], dtype=np.int64)
    use_locking = True
    name = "adadelta_6"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int64),
        "accum": tf.Variable(accum, dtype=tf.int64),
        "accum_update": tf.Variable(accum_update, dtype=tf.int64),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int64),
        "rho": tf.convert_to_tensor(rho, dtype=tf.int64),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.int64),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([1.0, 2.0], dtype=np.float32)
    accum = np.array([0.1, 0.2], dtype=np.float32)
    accum_update = np.array([0.3, 0.4], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(0.0, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    use_locking = False
    name = "adadelta_7"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "accum": tf.Variable(accum, dtype=tf.float32),
        "accum_update": tf.Variable(accum_update, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = np.array(1, dtype=np.int32)
    accum = np.array(0, dtype=np.int32)
    accum_update = np.array(0, dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(1, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array(1, dtype=np.int32)
    use_locking = True
    name = "adadelta_8"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int32),
        "accum": tf.Variable(accum, dtype=tf.int32),
        "accum_update": tf.Variable(accum_update, dtype=tf.int32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.int32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.int32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    use_locking = True
    name = "adadelta_9"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "accum": tf.Variable(accum, dtype=tf.float32),
        "accum_update": tf.Variable(accum_update, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([1, 2], dtype=np.int32)
    accum = np.array([0, 0], dtype=np.int32)
    accum_update = np.array([0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(1, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 1], dtype=np.int32)
    use_locking = False
    name = "adadelta_10"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int32),
        "accum": tf.Variable(accum, dtype=tf.int32),
        "accum_update": tf.Variable(accum_update, dtype=tf.int32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.int32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.int32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int32),
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
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdadelta'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyAdadelta', generated_inputs['tf.raw_ops.ApplyAdadelta'], lib="tf", suffix=0)
