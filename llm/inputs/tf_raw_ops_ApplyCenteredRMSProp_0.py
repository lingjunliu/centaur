
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_centered_rms_prop_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mom = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = False
    name = "CenteredRMSProp1"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    ms = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float64)
    mom = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    rho = np.array(0.95, dtype=np.float64)
    momentum = np.array(0.9, dtype=np.float64)
    epsilon = np.array(1e-8, dtype=np.float64)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    use_locking = True
    name = "CenteredRMSProp2"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 3
    var = np.array([1, 2, 3], dtype=np.int32)
    mg = np.array([0, 0, 0], dtype=np.int32)
    ms = np.array([1, 1, 1], dtype=np.int32)
    mom = np.array([0, 0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(0, dtype=np.int32)
    momentum = np.array(0, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    use_locking = False
    name = "CenteredRMSProp3"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

     # Input 4
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    mg = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mom = np.array([-0.7, -0.8, -0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    use_locking = False
    name = "CenteredRMSProp4"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 5
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64)
    mg = np.array([[0.1j, 0.2j], [0.3j, 0.4j]], dtype=np.complex64)
    ms = np.array([[0.4+0.1j, 0.5+0.2j], [0.6+0.3j, 0.7+0.4j]], dtype=np.complex64)
    mom = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.complex64)
    lr = np.array(0.001, dtype=np.complex64)
    rho = np.array(0.95, dtype=np.complex64)
    momentum = np.array(0.9, dtype=np.complex64)
    epsilon = np.array(1e-8, dtype=np.complex64)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.complex64)
    use_locking = True
    name = "CenteredRMSProp5"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 6
    var = np.array([1, 2, 3], dtype=np.int64)
    mg = np.array([0, 0, 0], dtype=np.int64)
    ms = np.array([1, 1, 1], dtype=np.int64)
    mom = np.array([0, 0, 0], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    rho = np.array(0, dtype=np.int64)
    momentum = np.array(0, dtype=np.int64)
    epsilon = np.array(1, dtype=np.int64)
    grad = np.array([1, 1, 1], dtype=np.int64)
    use_locking = False
    name = "CenteredRMSProp6"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 7
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mom = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(0.0, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    use_locking = False
    name = "CenteredRMSProp7"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 8
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mom = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    use_locking = False
    name = "CenteredRMSProp8"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

    # Input 9
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    ms = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float64)
    mom = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    rho = np.array(0.95, dtype=np.float64)
    momentum = np.array(0.9, dtype=np.float64)
    epsilon = np.array(0.0, dtype=np.float64)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    use_locking = True
    name = "CenteredRMSProp9"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)

     # Input 10
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    mg = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mom = np.array([-0.7, -0.8, -0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([-0.0, -0.0, -0.0], dtype=np.float32)
    use_locking = False
    name = "CenteredRMSProp10"
    input_dict = {"var": var, "mg": mg, "ms": ms, "mom": mom, "lr": lr, "rho": rho, "momentum": momentum, "epsilon": epsilon, "grad": grad, "use_locking": use_locking, "name": name}
    list_of_inputs.append(input_dict)


    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_apply_centered_rms_prop_inputs()
generated_inputs["tf.raw_ops.ApplyCenteredRMSProp"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.ApplyCenteredRMSProp"].append({
        'var': tf.Variable(np.array(input_dict['var'])),
        'mg': tf.Variable(np.array(input_dict['mg'])),
        'ms': tf.Variable(np.array(input_dict['ms'])),
        'mom': tf.Variable(np.array(input_dict['mom'])),
        'lr': input_dict['lr'],
        'rho': input_dict['rho'],
        'momentum': input_dict['momentum'],
        'epsilon': input_dict['epsilon'],
        'grad': input_dict['grad'],
        'use_locking': input_dict['use_locking'],
        'name': input_dict['name'],
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyCenteredRMSProp'.")

check_valid('tf.raw_ops.ApplyCenteredRMSProp', generated_inputs['tf.raw_ops.ApplyCenteredRMSProp'], lib="tf", suffix=0)
