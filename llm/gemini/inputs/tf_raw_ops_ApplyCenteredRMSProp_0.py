
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
    grad = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    ms = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
    mom = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32)
    use_locking = True
    name = "test_op"

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1.0], dtype=np.float64)
    mg = np.array([0.1], dtype=np.float64)
    ms = np.array([0.4], dtype=np.float64)
    mom = np.array([0.7], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    momentum = np.array(0.0, dtype=np.float64)
    epsilon = np.array(1e-7, dtype=np.float64)
    grad = np.array([0.1], dtype=np.float64)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([1, 2, 3], dtype=np.int32)
    mg = np.array([0, 0, 0], dtype=np.int32)
    ms = np.array([0, 0, 0], dtype=np.int32)
    mom = np.array([0, 0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(0, dtype=np.int32)
    momentum = np.array(0, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    mg = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mom = np.array([-0.7, -0.8, -0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([-0.1, -0.1, -0.1], dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mom = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Complex type
    var = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j], dtype=np.complex64)
    mg = np.array([0.1 + 0.1j, 0.2 + 0.2j, 0.3 + 0.3j], dtype=np.complex64)
    ms = np.array([0.4 + 0.4j, 0.5 + 0.5j, 0.6 + 0.6j], dtype=np.complex64)
    mom = np.array([0.7 + 0.7j, 0.8 + 0.8j, 0.9 + 0.9j], dtype=np.complex64)
    lr = np.array(0.01 + 0j, dtype=np.complex64)
    rho = np.array(0.9 + 0j, dtype=np.complex64)
    momentum = np.array(0.0 + 0j, dtype=np.complex64)
    epsilon = np.array(1e-7 + 0j, dtype=np.complex64)
    grad = np.array([0.1 + 0.1j, 0.1 + 0.1j, 0.1 + 0.1j], dtype=np.complex64)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - different shapes
    var = np.array(1.0, dtype=np.float32)
    mg = np.array(0.1, dtype=np.float32)
    ms = np.array(0.4, dtype=np.float32)
    mom = np.array(0.7, dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array(0.1, dtype=np.float32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - different values
    var = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    mg = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    ms = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mom = np.array([1.5, 2.0, 2.5], dtype=np.float32)
    lr = np.array(0.05, dtype=np.float32)
    rho = np.array(0.8, dtype=np.float32)
    momentum = np.array(0.1, dtype=np.float32)
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    use_locking = True
    name = "another_test_op"

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - 0 grad
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
    name = None

    input_dict = {
        "var": var,
        "mg": mg,
        "ms": ms,
        "mom": mom,
        "lr": lr,
        "rho": rho,
        "momentum": momentum,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyCenteredRMSProp"] = tf_raw_ops_apply_centered_rms_prop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyCenteredRMSProp'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyCenteredRMSProp', generated_inputs['tf.raw_ops.ApplyCenteredRMSProp'], lib="tf", suffix=0)
