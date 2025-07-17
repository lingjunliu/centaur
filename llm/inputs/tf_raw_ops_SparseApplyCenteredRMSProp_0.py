
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyCenteredRMSProp_inputs():
    list_of_inputs = []

    # Input 1, valid
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    ms = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    mom = np.array([[0.9, 1.0], [1.1, 1.2]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_1"

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
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, different data type
    var = np.array([[1, 2], [3, 4]], dtype=np.float64)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    ms = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    mom = np.array([[0.9, 1.0], [1.1, 1.2]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    momentum = np.array(0.0, dtype=np.float64)
    epsilon = np.array(1e-7, dtype=np.float64)
    grad = np.array([[0.2, 0.3]], dtype=np.float64)
    indices = np.array([0], dtype=np.int64)
    use_locking = True
    name = "sparse_apply_centered_rmsprop_2"

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
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, different values
    var = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    mg = np.array([[0.15, 0.25], [0.35, 0.45]], dtype=np.float32)
    ms = np.array([[0.55, 0.65], [0.75, 0.85]], dtype=np.float32)
    mom = np.array([[0.95, 1.05], [1.15, 1.25]], dtype=np.float32)
    lr = np.array(0.02, dtype=np.float32)
    rho = np.array(0.8, dtype=np.float32)
    momentum = np.array(0.1, dtype=np.float32)
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([[0.25, 0.35]], dtype=np.float32)
    indices = np.array([1], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_3"

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
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4, valid, multi-dimensional grad and indices
    var = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    ms = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    mom = np.array([[0.9, 1.0], [1.1, 1.2], [1.3, 1.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_4"

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
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, different shapes
    var = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    ms = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=np.float32)
    mom = np.array([[0.9, 1.0, 1.1], [1.2, 1.3, 1.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([[0.2, 0.3, 0.4]], dtype=np.float32)
    indices = np.array([1], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_5"

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
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, momentum > 0
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    ms = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    mom = np.array([[0.9, 1.0], [1.1, 1.2]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.1, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_6"

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
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, larger learning rate
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    ms = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    mom = np.array([[0.9, 1.0], [1.1, 1.2]], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_7"

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
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, different rho
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    ms = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    mom = np.array([[0.9, 1.0], [1.1, 1.2]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.5, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([[0.2, 0.3]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_8"

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
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    var = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    ms = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]], dtype=np.float32)
    mom = np.array([[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-7, dtype=np.float32)
    grad = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
    indices = np.array([0, 2, 3], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_centered_rmsprop_10"

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
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyCenteredRMSProp"] = tf_raw_ops_SparseApplyCenteredRMSProp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyCenteredRMSProp'.")

check_valid('tf.raw_ops.SparseApplyCenteredRMSProp', generated_inputs['tf.raw_ops.SparseApplyCenteredRMSProp'], lib="tf", suffix=0)
