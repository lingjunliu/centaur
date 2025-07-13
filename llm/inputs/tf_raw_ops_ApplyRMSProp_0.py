
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_rms_prop_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ms = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mom = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-10, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    use_locking = False
    name = "rmsprop_1"

    input_dict = {
        "var": var,
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
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    ms = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    mom = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    momentum = np.array(0.0, dtype=np.float64)
    epsilon = np.array(1e-10, dtype=np.float64)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    use_locking = True
    name = "rmsprop_2"

    input_dict = {
        "var": var,
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
    var = np.array([1, 2, 3], dtype=np.int32)
    ms = np.array([1, 2, 3], dtype=np.int32)
    mom = np.array([1, 2, 3], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(1, dtype=np.int32)
    momentum = np.array(1, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    use_locking = False
    name = "rmsprop_3"

    input_dict = {
        "var": var,
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

    # Input 4: Multi-dimensional tensors
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ms = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    mom = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    epsilon = np.array(1e-10, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    use_locking = False
    name = "rmsprop_4"

    input_dict = {
        "var": var,
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
    
    # Input 5: Different parameters
    var = np.array([0.5, -1.0], dtype=np.float32)
    ms = np.array([0.2, 0.4], dtype=np.float32)
    mom = np.array([0.1, -0.05], dtype=np.float32)
    lr = np.array(0.05, dtype=np.float32)
    rho = np.array(0.95, dtype=np.float32)
    momentum = np.array(0.1, dtype=np.float32)
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([0.3, -0.2], dtype=np.float32)
    use_locking = True
    name = "rmsprop_5"
    
    input_dict = {
        "var": var,
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

    # Input 6: int64 tensors
    var = np.array([1, 2, 3], dtype=np.int64)
    ms = np.array([1, 2, 3], dtype=np.int64)
    mom = np.array([1, 2, 3], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    rho = np.array(1, dtype=np.int64)
    momentum = np.array(1, dtype=np.int64)
    epsilon = np.array(1, dtype=np.int64)
    grad = np.array([1, 1, 1], dtype=np.int64)
    use_locking = False
    name = "rmsprop_6"

    input_dict = {
        "var": var,
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
    
    # Input 7: Different values for rho and momentum
    var = np.array([0.5, -1.0], dtype=np.float32)
    ms = np.array([0.2, 0.4], dtype=np.float32)
    mom = np.array([0.1, -0.05], dtype=np.float32)
    lr = np.array(0.05, dtype=np.float32)
    rho = np.array(0.8, dtype=np.float32)  # lower rho
    momentum = np.array(0.5, dtype=np.float32)  # higher momentum
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([0.3, -0.2], dtype=np.float32)
    use_locking = True
    name = "rmsprop_7"
    
    input_dict = {
        "var": var,
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
    
    # Input 8: uint8
    var = np.array([1, 2, 3], dtype=np.uint8)
    ms = np.array([1, 2, 3], dtype=np.uint8)
    mom = np.array([1, 2, 3], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    rho = np.array(1, dtype=np.uint8)
    momentum = np.array(1, dtype=np.uint8)
    epsilon = np.array(1, dtype=np.uint8)
    grad = np.array([1, 1, 1], dtype=np.uint8)
    use_locking = False
    name = "rmsprop_8"

    input_dict = {
        "var": var,
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
        
    # Input 9: half
    var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    ms = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    mom = np.array([0.01, 0.02, 0.03], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    rho = np.array(0.9, dtype=np.float16)
    momentum = np.array(0.0, dtype=np.float16)
    epsilon = np.array(1e-10, dtype=np.float16)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float16)
    use_locking = False
    name = "rmsprop_9"

    input_dict = {
        "var": var,
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

    # Input 10: uint32
    var = np.array([1, 2, 3], dtype=np.uint32)
    ms = np.array([1, 2, 3], dtype=np.uint32)
    mom = np.array([1, 2, 3], dtype=np.uint32)
    lr = np.array(1, dtype=np.uint32)
    rho = np.array(1, dtype=np.uint32)
    momentum = np.array(1, dtype=np.uint32)
    epsilon = np.array(1, dtype=np.uint32)
    grad = np.array([1, 1, 1], dtype=np.uint32)
    use_locking = False
    name = "rmsprop_10"

    input_dict = {
        "var": var,
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
inputs = tf_raw_ops_apply_rms_prop_inputs()
for i in range(len(inputs)):
  for key in inputs[i]:
    inputs[i][key] = tf.convert_to_tensor(np.array(inputs[i][key]))

generated_inputs["tf.raw_ops.ApplyRMSProp"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyRMSProp'.")

check_valid('tf.raw_ops.ApplyRMSProp', generated_inputs['tf.raw_ops.ApplyRMSProp'], lib="tf", suffix=0)
