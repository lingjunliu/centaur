
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_SparseApplyCenteredRMSProp_inputs():
    list_of_inputs = []

    # Input 1, valid
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ms = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    mom = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.0, 0.4], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_1"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 2, valid, different types
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    mg = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ms = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    mom = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    momentum = np.array(0.0, dtype=np.float64)
    epsilon = np.array(1e-7, dtype=np.float64)
    grad = np.array([0.1, 0.2, 0.0, 0.4], dtype=np.float64)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    use_locking = True
    name = "sparse_apply_centered_rmsprop_2"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float64),
        "mg": tf.Variable(mg, dtype=tf.float64),
        "ms": tf.Variable(ms, dtype=tf.float64),
        "mom": tf.Variable(mom, dtype=tf.float64),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float64),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float64),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float64),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float64),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float64),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 3, valid, different values
    var = np.array([1.0, 2.0], dtype=np.float32)
    mg = np.array([0.0, 0.0], dtype=np.float32)
    ms = np.array([0.0, 0.0], dtype=np.float32)
    mom = np.array([0.0, 0.0], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    rho = np.array(0.99, dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    epsilon = np.array(0.001, dtype=np.float32)
    grad = np.array([0.5, 0.5], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_3"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 4, valid
    var = np.array([1.0], dtype=np.float32)
    mg = np.array([0.1], dtype=np.float32)
    ms = np.array([0.5], dtype=np.float32)
    mom = np.array([0.01], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_4"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 5, valid
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ms = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    mom = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_5"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 6, valid, int32 var
    var = np.array([1, 2, 3, 4], dtype=np.int32)
    mg = np.array([0, 0, 0, 0], dtype=np.int32)
    ms = np.array([0, 0, 0, 0], dtype=np.int32)
    mom = np.array([0, 0, 0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(0, dtype=np.int32)
    momentum = np.array(0, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([0, 0, 0, 0], dtype=np.int32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_6"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int32),
        "mg": tf.Variable(mg, dtype=tf.int32),
        "ms": tf.Variable(ms, dtype=tf.int32),
        "mom": tf.Variable(mom, dtype=tf.int32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.int32),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.int32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.int32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 7, valid, different indices
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ms = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    mom = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.0, 0.4], dtype=np.float32)
    indices = np.array([3, 2, 1, 0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_7"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 8, valid, small values
    var = np.array([1e-8, 2e-8, 3e-8, 4e-8], dtype=np.float32)
    mg = np.array([0.1e-8, 0.2e-8, 0.3e-8, 0.4e-8], dtype=np.float32)
    ms = np.array([0.5e-8, 0.6e-8, 0.7e-8, 0.8e-8], dtype=np.float32)
    mom = np.array([0.01e-8, 0.02e-8, 0.03e-8, 0.04e-8], dtype=np.float32)
    lr = np.array(0.01e-8, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-9, dtype=np.float32)
    grad = np.array([0.1e-8, 0.2e-8, 0.0, 0.4e-8], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_8"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 9, valid, use locking true
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ms = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    mom = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.0, 0.4], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    use_locking = True
    name = "sparse_apply_centered_rmsprop_9"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)

    # Input 10, valid, different name
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mg = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ms = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    mom = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.0, 0.4], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    use_locking = False
    name = "another_name"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "rho": tf.convert_to_tensor(rho, dtype=tf.float32),
        "momentum": tf.convert_to_tensor(momentum, dtype=tf.float32),
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyCenteredRMSProp"] = tf_raw_ops_SparseApplyCenteredRMSProp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyCenteredRMSProp'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseApplyCenteredRMSProp', generated_inputs['tf.raw_ops.SparseApplyCenteredRMSProp'], lib="tf", suffix=0)
