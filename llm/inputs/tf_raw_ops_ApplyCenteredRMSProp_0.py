
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ApplyCenteredRMSProp_inputs():
    list_of_inputs = []

    # Input 1: Basic test with float32
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mom = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    input_dict = {
        "var": tf.Variable(var).numpy(),
        "mg": tf.Variable(mg).numpy(),
        "ms": tf.Variable(ms).numpy(),
        "mom": tf.Variable(mom).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32).numpy(),
        "grad": grad,
        "use_locking": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    ms = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
    mom = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    input_dict = {
        "var": tf.Variable(var).numpy(),
        "mg": tf.Variable(mg).numpy(),
        "ms": tf.Variable(ms).numpy(),
        "mom": tf.Variable(mom).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32).numpy(),
        "grad": grad,
        "use_locking": True,
        "name": "test_name"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different learning rate and momentum
    var = np.array([1.0, 2.0], dtype=np.float32)
    mg = np.array([0.1, 0.2], dtype=np.float32)
    ms = np.array([0.4, 0.5], dtype=np.float32)
    mom = np.array([0.7, 0.8], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1, 0.2], dtype=np.float32)

    input_dict = {
        "var": tf.Variable(var).numpy(),
        "mg": tf.Variable(mg).numpy(),
        "ms": tf.Variable(ms).numpy(),
        "mom": tf.Variable(mom).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32).numpy(),
        "grad": grad,
        "use_locking": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger values for gradient
    var = np.array([1.0, 2.0], dtype=np.float32)
    mg = np.array([0.1, 0.2], dtype=np.float32)
    ms = np.array([0.4, 0.5], dtype=np.float32)
    mom = np.array([0.7, 0.8], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([1.0, 2.0], dtype=np.float32)

    input_dict = {
        "var": tf.Variable(var).numpy(),
        "mg": tf.Variable(mg).numpy(),
        "ms": tf.Variable(ms).numpy(),
        "mom": tf.Variable(mom).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32).numpy(),
        "grad": grad,
        "use_locking": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different epsilon value
    var = np.array([1.0, 2.0], dtype=np.float32)
    mg = np.array([0.1, 0.2], dtype=np.float32)
    ms = np.array([0.4, 0.5], dtype=np.float32)
    mom = np.array([0.7, 0.8], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-3, dtype=np.float32)
    grad = np.array([0.1, 0.2], dtype=np.float32)

    input_dict = {
        "var": tf.Variable(var).numpy(),
        "mg": tf.Variable(mg).numpy(),
        "ms": tf.Variable(ms).numpy(),
        "mom": tf.Variable(mom).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32).numpy(),
        "grad": grad,
        "use_locking": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative gradient values
    var = np.array([1.0, 2.0], dtype=np.float32)
    mg = np.array([0.1, 0.2], dtype=np.float32)
    ms = np.array([0.4, 0.5], dtype=np.float32)
    mom = np.array([0.7, 0.8], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([-0.1, -0.2], dtype=np.float32)

    input_dict = {
        "var": tf.Variable(var).numpy(),
        "mg": tf.Variable(mg).numpy(),
        "ms": tf.Variable(ms).numpy(),
        "mom": tf.Variable(mom).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32).numpy(),
        "grad": grad,
        "use_locking": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    mom = np.array([0.7, 0.8, 0.9], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    momentum = np.array(0.0, dtype=np.float64)
    epsilon = np.array(1e-7, dtype=np.float64)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float64).numpy(),
        "mg": tf.Variable(mg, dtype=tf.float64).numpy(),
        "ms": tf.Variable(ms, dtype=tf.float64).numpy(),
        "mom": tf.Variable(mom, dtype=tf.float64).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float64).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float64).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float64).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float64).numpy(),
        "grad": grad,
        "use_locking": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32
    var = np.array([1, 2, 3], dtype=np.int32)
    mg = np.array([1, 2, 3], dtype=np.int32)
    ms = np.array([4, 5, 6], dtype=np.int32)
    mom = np.array([7, 8, 9], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(9, dtype=np.int32)
    momentum = np.array(0, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 2, 3], dtype=np.int32)

    input_dict = {
        "var": tf.Variable(var).numpy(),
        "mg": tf.Variable(mg).numpy(),
        "ms": tf.Variable(ms).numpy(),
        "mom": tf.Variable(mom).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int32).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.int32).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.int32).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.int32).numpy(),
        "grad": grad,
        "use_locking": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: higher rank
    var = np.random.rand(2,3,4).astype(np.float32)
    mg = np.random.rand(2,3,4).astype(np.float32)
    ms = np.random.rand(2,3,4).astype(np.float32)
    mom = np.random.rand(2,3,4).astype(np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.random.rand(2,3,4).astype(np.float32)

    input_dict = {
        "var": tf.Variable(var).numpy(),
        "mg": tf.Variable(mg).numpy(),
        "ms": tf.Variable(ms).numpy(),
        "mom": tf.Variable(mom).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32).numpy(),
        "grad": grad,
        "use_locking": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8
    var = np.array([1, 2, 3], dtype=np.uint8)
    mg = np.array([1, 2, 3], dtype=np.uint8)
    ms = np.array([4, 5, 6], dtype=np.uint8)
    mom = np.array([7, 8, 9], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    rho = np.array(9, dtype=np.uint8)
    momentum = np.array(0, dtype=np.uint8)
    epsilon = np.array(1, dtype=np.uint8)
    grad = np.array([1, 2, 3], dtype=np.uint8)

    input_dict = {
        "var": tf.Variable(var).numpy(),
        "mg": tf.Variable(mg).numpy(),
        "ms": tf.Variable(ms).numpy(),
        "mom": tf.Variable(mom).numpy(),
        "lr": tf.convert_to_tensor(lr, dtype=tf.uint8).numpy(),
        "rho": tf.convert_to_tensor(rho, dtype=tf.uint8).numpy(),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.uint8).numpy(),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.uint8).numpy(),
        "grad": grad,
        "use_locking": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyCenteredRMSProp"] = tf_raw_ops_ApplyCenteredRMSProp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyCenteredRMSProp'.")

check_valid('tf.raw_ops.ApplyCenteredRMSProp', generated_inputs['tf.raw_ops.ApplyCenteredRMSProp'], lib="tf", suffix=0)
