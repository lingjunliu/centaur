
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
    grad = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    use_locking = False
    name = "centered_rms_prop_1"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.constant(lr, dtype=tf.float32),
        "rho": tf.constant(rho, dtype=tf.float32),
        "momentum": tf.constant(momentum, dtype=tf.float32),
        "epsilon": tf.constant(epsilon, dtype=tf.float32),
        "grad": tf.constant(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    ms = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
    mom = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    lr = np.array(0.001, dtype=np.float32)
    rho = np.array(0.95, dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    use_locking = True
    name = "centered_rms_prop_2"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.constant(lr, dtype=tf.float32),
        "rho": tf.constant(rho, dtype=tf.float32),
        "momentum": tf.constant(momentum, dtype=tf.float32),
        "epsilon": tf.constant(epsilon, dtype=tf.float32),
        "grad": tf.constant(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 3
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    mom = np.array([0.7, 0.8, 0.9], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    momentum = np.array(0.0, dtype=np.float64)
    epsilon = np.array(1e-7, dtype=np.float64)
    grad = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    use_locking = False
    name = "centered_rms_prop_3"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float64),
        "mg": tf.Variable(mg, dtype=tf.float64),
        "ms": tf.Variable(ms, dtype=tf.float64),
        "mom": tf.Variable(mom, dtype=tf.float64),
        "lr": tf.constant(lr, dtype=tf.float64),
        "rho": tf.constant(rho, dtype=tf.float64),
        "momentum": tf.constant(momentum, dtype=tf.float64),
        "epsilon": tf.constant(epsilon, dtype=tf.float64),
        "grad": tf.constant(grad, dtype=tf.float64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    mg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    ms = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float64)
    mom = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    rho = np.array(0.95, dtype=np.float64)
    momentum = np.array(0.9, dtype=np.float64)
    epsilon = np.array(1e-8, dtype=np.float64)
    grad = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    use_locking = True
    name = "centered_rms_prop_4"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float64),
        "mg": tf.Variable(mg, dtype=tf.float64),
        "ms": tf.Variable(ms, dtype=tf.float64),
        "mom": tf.Variable(mom, dtype=tf.float64),
        "lr": tf.constant(lr, dtype=tf.float64),
        "rho": tf.constant(rho, dtype=tf.float64),
        "momentum": tf.constant(momentum, dtype=tf.float64),
        "epsilon": tf.constant(epsilon, dtype=tf.float64),
        "grad": tf.constant(grad, dtype=tf.float64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 (int32)
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
    name = "centered_rms_prop_5"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int32),
        "mg": tf.Variable(mg, dtype=tf.int32),
        "ms": tf.Variable(ms, dtype=tf.int32),
        "mom": tf.Variable(mom, dtype=tf.int32),
        "lr": tf.constant(lr, dtype=tf.int32),
        "rho": tf.constant(rho, dtype=tf.int32),
        "momentum": tf.constant(momentum, dtype=tf.int32),
        "epsilon": tf.constant(epsilon, dtype=tf.int32),
        "grad": tf.constant(grad, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 (int64)
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
    name = "centered_rms_prop_6"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.int64),
        "mg": tf.Variable(mg, dtype=tf.int64),
        "ms": tf.Variable(ms, dtype=tf.int64),
        "mom": tf.Variable(mom, dtype=tf.int64),
        "lr": tf.constant(lr, dtype=tf.int64),
        "rho": tf.constant(rho, dtype=tf.int64),
        "momentum": tf.constant(momentum, dtype=tf.int64),
        "epsilon": tf.constant(epsilon, dtype=tf.int64),
        "grad": tf.constant(grad, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 (bfloat16)
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32) #np.bfloat16 is not directly supported in numpy, using float32 instead
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float32) #np.bfloat16 is not directly supported in numpy, using float32 instead
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float32) #np.bfloat16 is not directly supported in numpy, using float32 instead
    mom = np.array([0.7, 0.8, 0.9], dtype=np.float32) #np.bfloat16 is not directly supported in numpy, using float32 instead
    lr = np.array(0.01, dtype=np.float32) #np.bfloat16 is not directly supported in numpy, using float32 instead
    rho = np.array(0.9, dtype=np.float32) #np.bfloat16 is not directly supported in numpy, using float32 instead
    momentum = np.array(0.0, dtype=np.float32) #np.bfloat16 is not directly supported in numpy, using float32 instead
    epsilon = np.array(1e-7, dtype=np.float32) #np.bfloat16 is not directly supported in numpy, using float32 instead
    grad = np.array([0.01, 0.02, 0.03], dtype=np.float32) #np.bfloat16 is not directly supported in numpy, using float32 instead
    use_locking = False
    name = "centered_rms_prop_7"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float32),
        "mg": tf.Variable(mg, dtype=tf.float32),
        "ms": tf.Variable(ms, dtype=tf.float32),
        "mom": tf.Variable(mom, dtype=tf.float32),
        "lr": tf.constant(lr, dtype=tf.float32),
        "rho": tf.constant(rho, dtype=tf.float32),
        "momentum": tf.constant(momentum, dtype=tf.float32),
        "epsilon": tf.constant(epsilon, dtype=tf.float32),
        "grad": tf.constant(grad, dtype=tf.float32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (half)
    var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    mg = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    ms = np.array([0.4, 0.5, 0.6], dtype=np.float16)
    mom = np.array([0.7, 0.8, 0.9], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    rho = np.array(0.9, dtype=np.float16)
    momentum = np.array(0.0, dtype=np.float16)
    epsilon = np.array(1e-7, dtype=np.float16)
    grad = np.array([0.01, 0.02, 0.03], dtype=np.float16)
    use_locking = False
    name = "centered_rms_prop_8"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.float16),
        "mg": tf.Variable(mg, dtype=tf.float16),
        "ms": tf.Variable(ms, dtype=tf.float16),
        "mom": tf.Variable(mom, dtype=tf.float16),
        "lr": tf.constant(lr, dtype=tf.float16),
        "rho": tf.constant(rho, dtype=tf.float16),
        "momentum": tf.constant(momentum, dtype=tf.float16),
        "epsilon": tf.constant(epsilon, dtype=tf.float16),
        "grad": tf.constant(grad, dtype=tf.float16),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 (complex64)
    var = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], dtype=np.complex64)
    mg = np.array([0.1 + 0.1j, 0.2 + 0.2j, 0.3 + 0.3j], dtype=np.complex64)
    ms = np.array([0.4 + 0.4j, 0.5 + 0.5j, 0.6 + 0.6j], dtype=np.complex64)
    mom = np.array([0.7 + 0.7j, 0.8 + 0.8j, 0.9 + 0.9j], dtype=np.complex64)
    lr = np.array(0.01 + 0.0j, dtype=np.complex64)
    rho = np.array(0.9 + 0.0j, dtype=np.complex64)
    momentum = np.array(0.0 + 0.0j, dtype=np.complex64)
    epsilon = np.array(1e-7 + 0.0j, dtype=np.complex64)
    grad = np.array([0.01 + 0.0j, 0.02 + 0.0j, 0.03 + 0.0j], dtype=np.complex64)
    use_locking = False
    name = "centered_rms_prop_9"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.complex64),
        "mg": tf.Variable(mg, dtype=tf.complex64),
        "ms": tf.Variable(ms, dtype=tf.complex64),
        "mom": tf.Variable(mom, dtype=tf.complex64),
        "lr": tf.constant(lr, dtype=tf.complex64),
        "rho": tf.constant(rho, dtype=tf.complex64),
        "momentum": tf.constant(momentum, dtype=tf.complex64),
        "epsilon": tf.constant(epsilon, dtype=tf.complex64),
        "grad": tf.constant(grad, dtype=tf.complex64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 (complex128)
    var = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], dtype=np.complex128)
    mg = np.array([0.1 + 0.1j, 0.2 + 0.2j, 0.3 + 0.3j], dtype=np.complex128)
    ms = np.array([0.4 + 0.4j, 0.5 + 0.5j, 0.6 + 0.6j], dtype=np.complex128)
    mom = np.array([0.7 + 0.7j, 0.8 + 0.8j, 0.9 + 0.9j], dtype=np.complex128)
    lr = np.array(0.01 + 0.0j, dtype=np.complex128)
    rho = np.array(0.9 + 0.0j, dtype=np.complex128)
    momentum = np.array(0.0 + 0.0j, dtype=np.complex128)
    epsilon = np.array(1e-7 + 0.0j, dtype=np.complex128)
    grad = np.array([0.01 + 0.0j, 0.02 + 0.0j, 0.03 + 0.0j], dtype=np.complex128)
    use_locking = False
    name = "centered_rms_prop_10"

    input_dict = {
        "var": tf.Variable(var, dtype=tf.complex128),
        "mg": tf.Variable(mg, dtype=tf.complex128),
        "ms": tf.Variable(ms, dtype=tf.complex128),
        "mom": tf.Variable(mom, dtype=tf.complex128),
        "lr": tf.constant(lr, dtype=tf.complex128),
        "rho": tf.constant(rho, dtype=tf.complex128),
        "momentum": tf.constant(momentum, dtype=tf.complex128),
        "epsilon": tf.constant(epsilon, dtype=tf.complex128),
        "grad": tf.constant(grad, dtype=tf.complex128),
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
    
    print("Valid")

if 'tf.raw_ops.ApplyCenteredRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyCenteredRMSProp'.")

check_valid('tf.raw_ops.ApplyCenteredRMSProp', generated_inputs['tf.raw_ops.ApplyCenteredRMSProp'], lib="tf", suffix=0)
