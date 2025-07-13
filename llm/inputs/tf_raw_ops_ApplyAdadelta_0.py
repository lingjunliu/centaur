
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adadelta_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    accum_update = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    use_locking = False
    name = "adadelta_1"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

    # Input 2: float64, different values
    var = np.array([-1.0, 2.5, -3.2], dtype=np.float64)
    accum = np.array([0.5, 0.25, 0.1], dtype=np.float64)
    accum_update = np.array([0.2, 0.1, 0.05], dtype=np.float64)
    lr = np.array(0.005, dtype=np.float64)
    rho = np.array(0.95, dtype=np.float64)
    epsilon = np.array(1e-8, dtype=np.float64)
    grad = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    use_locking = True
    name = "adadelta_2"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

    # Input 3: int32, positive values
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 1, 1], dtype=np.int32)
    accum_update = np.array([1, 1, 1], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(0, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 2, 3], dtype=np.int32)
    use_locking = False
    name = "adadelta_3"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

    # Input 4: int64, negative values
    var = np.array([-1, -2, -3], dtype=np.int64)
    accum = np.array([1, 1, 1], dtype=np.int64)
    accum_update = np.array([1, 1, 1], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    rho = np.array(0, dtype=np.int64)
    epsilon = np.array(1, dtype=np.int64)
    grad = np.array([-1, -2, -3], dtype=np.int64)
    use_locking = True
    name = "adadelta_4"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

     # Input 5: float32, zero values
    var = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_update = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lr = np.array(0.0, dtype=np.float32)
    rho = np.array(0.0, dtype=np.float32)
    epsilon = np.array(0.0, dtype=np.float32)
    grad = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    use_locking = False
    name = "adadelta_5"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

    # Input 6: float32, larger values
    var = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    accum = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    accum_update = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    lr = np.array(1.0, dtype=np.float32)
    rho = np.array(0.5, dtype=np.float32)
    epsilon = np.array(0.00001, dtype=np.float32)
    grad = np.array([500.0, 600.0, 700.0], dtype=np.float32)
    use_locking = True
    name = "adadelta_6"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

    # Input 7: float32, different shapes
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    use_locking = False
    name = "adadelta_7"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

    # Input 8: float64, 3D tensor
    var = np.random.rand(2, 3, 4).astype(np.float64)
    accum = np.random.rand(2, 3, 4).astype(np.float64)
    accum_update = np.random.rand(2, 3, 4).astype(np.float64)
    lr = np.array(0.001, dtype=np.float64)
    rho = np.array(0.99, dtype=np.float64)
    epsilon = np.array(1e-7, dtype=np.float64)
    grad = np.random.rand(2, 3, 4).astype(np.float64)
    use_locking = True
    name = "adadelta_8"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

    # Input 9: int32, different dimensions
    var = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    accum = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=np.int32)
    accum_update = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(0, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    use_locking = False
    name = "adadelta_9"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

    # Input 10: float32, with a small learning rate
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    accum_update = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    lr = np.array(1e-6, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    use_locking = True
    name = "adadelta_10"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append({"args": [], "kwargs": input_dict})

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
