
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyRMSProp_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ms = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    mom = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-10, dtype=np.float32)
    grad = np.array([[0.5, 0.6]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "rms_prop_1"

    input_dict = {
        "var": var,
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

    # Input 2
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ms = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mom = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    lr = np.array(0.001, dtype=np.float32)
    rho = np.array(0.95, dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([0.2, 0.4], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = True
    name = "rms_prop_2"

    input_dict = {
        "var": var,
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

    # Input 3
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    ms = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    mom = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    momentum = np.array(0.0, dtype=np.float64)
    epsilon = np.array(1e-10, dtype=np.float64)
    grad = np.array([[0.5, 0.6]], dtype=np.float64)
    indices = np.array([0], dtype=np.int64)
    use_locking = False
    name = "rms_prop_3"

    input_dict = {
        "var": var,
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

    # Input 4
    var = np.array([1, 2, 3], dtype=np.int32)
    ms = np.array([1, 1, 1], dtype=np.int32)
    mom = np.array([0, 0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(1, dtype=np.int32)
    momentum = np.array(0, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 1], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = True
    name = "rms_prop_4"

    input_dict = {
        "var": var,
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

    # Input 5
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ms = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mom = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    lr = np.array(-0.001, dtype=np.float32)
    rho = np.array(0.95, dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([-0.2, 0.4], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = True
    name = "rms_prop_5"

    input_dict = {
        "var": var,
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

     # Input 6
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64)
    ms = np.array([[0.1+1j, 0.2+2j], [0.3+3j, 0.4+4j]], dtype=np.complex64)
    mom = np.array([[0.01+1j, 0.02+2j], [0.03+3j, 0.04+4j]], dtype=np.complex64)
    lr = np.array(0.01+1j, dtype=np.complex64)
    rho = np.array(0.9+1j, dtype=np.complex64)
    momentum = np.array(0.0+1j, dtype=np.complex64)
    epsilon = np.array(1e-10+1j, dtype=np.complex64)
    grad = np.array([[0.5+1j, 0.6+2j]], dtype=np.complex64)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "rms_prop_6"

    input_dict = {
        "var": var,
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

    # Input 7
    #var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    #ms = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    #mom = np.array([0.01, 0.02, 0.03], dtype=np.float16)
    #lr = np.array(0.001, dtype=np.float16)
    #rho = np.array(0.95, dtype=np.float16)
    #momentum = np.array(0.9, dtype=np.float16)
    #epsilon = np.array(1e-8, dtype=np.float16)
    #grad = np.array([0.2, 0.4], dtype=np.float16)
    #indices = np.array([0, 2], dtype=np.int32)
    #use_locking = True
    #name = "rms_prop_7"

    #input_dict = {
    #    "var": var,
    #    "ms": ms,
    #    "mom": mom,
    #    "lr": lr,
    #    "rho": rho,
    #    "momentum": momentum,
    #    "epsilon": epsilon,
    #    "grad": grad,
    #    "indices": indices,
    #    "use_locking": use_locking,
    #    "name": name
    #}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = np.array([1, 2, 3], dtype=np.int64)
    ms = np.array([1, 1, 1], dtype=np.int64)
    mom = np.array([0, 0, 0], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    rho = np.array(1, dtype=np.int64)
    momentum = np.array(0, dtype=np.int64)
    epsilon = np.array(1, dtype=np.int64)
    grad = np.array([1, 1], dtype=np.int64)
    indices = np.array([0, 2], dtype=np.int64)
    use_locking = True
    name = "rms_prop_8"

    input_dict = {
        "var": var,
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

    # Input 9
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ms = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    mom = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-10, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = False
    name = "rms_prop_9"

    input_dict = {
        "var": var,
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
    
    # Input 10
    var = np.array([1, 2, 3], dtype=np.int16)
    ms = np.array([1, 1, 1], dtype=np.int16)
    mom = np.array([0, 0, 0], dtype=np.int16)
    lr = np.array(1, dtype=np.int16)
    rho = np.array(1, dtype=np.int16)
    momentum = np.array(0, dtype=np.int16)
    epsilon = np.array(1, dtype=np.int16)
    grad = np.array([1, 1], dtype=np.int16)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = True
    name = "rms_prop_10"

    input_dict = {
        "var": var,
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
generated_inputs["tf.raw_ops.SparseApplyRMSProp"] = tf_raw_ops_SparseApplyRMSProp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseApplyRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyRMSProp'.")

check_valid('tf.raw_ops.SparseApplyRMSProp', generated_inputs['tf.raw_ops.SparseApplyRMSProp'], lib="tf", suffix=0)
