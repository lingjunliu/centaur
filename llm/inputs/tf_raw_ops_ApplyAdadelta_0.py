
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adadelta_inputs():
    list_of_inputs = []

    # Input 1: Basic test case with float32
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    accum_update = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    use_locking = False
    name = "adadelta_1"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different learning rate and rho
    var = np.array([1.0, 2.0], dtype=np.float32)
    accum = np.array([0.1, 0.2], dtype=np.float32)
    accum_update = np.array([0.01, 0.02], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    rho = np.array(0.95, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    use_locking = True
    name = "adadelta_2"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher dimensional tensor
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    use_locking = False
    name = "adadelta_3"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4: int32 type
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 1, 1], dtype=np.int32)
    accum_update = np.array([1, 1, 1], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(1, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    use_locking = False
    name = "adadelta_4"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes for variable and gradient
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    accum_update = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    use_locking = False
    name = "adadelta_5"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  complex64
    var = np.array([1.0+1j, 2.0+2j, 3.0+3j], dtype=np.complex64)
    accum = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex64)
    accum_update = np.array([0.01+0.01j, 0.02+0.02j, 0.03+0.03j], dtype=np.complex64)
    lr = np.array(0.01+0j, dtype=np.complex64)
    rho = np.array(0.9+0j, dtype=np.complex64)
    epsilon = np.array(1e-6+0j, dtype=np.complex64)
    grad = np.array([0.5+0.5j, 0.6+0.6j, 0.7+0.7j], dtype=np.complex64)
    use_locking = False
    name = "adadelta_6"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 type
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    accum_update = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    epsilon = np.array(1e-6, dtype=np.float64)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    use_locking = False
    name = "adadelta_7"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative Gradients and initial values
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    accum = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    accum_update = np.array([-0.01, -0.02, -0.03], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([-0.5, -0.6, -0.7], dtype=np.float32)
    use_locking = False
    name = "adadelta_8"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8
    var = np.array([1, 2, 3], dtype=np.uint8)
    accum = np.array([1, 1, 1], dtype=np.uint8)
    accum_update = np.array([1, 1, 1], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    rho = np.array(1, dtype=np.uint8)
    epsilon = np.array(1, dtype=np.uint8)
    grad = np.array([1, 1, 1], dtype=np.uint8)
    use_locking = False
    name = "adadelta_9"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128
    var = np.array([1.0+1j, 2.0+2j, 3.0+3j], dtype=np.complex128)
    accum = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex128)
    accum_update = np.array([0.01+0.01j, 0.02+0.02j, 0.03+0.03j], dtype=np.complex128)
    lr = np.array(0.01+0j, dtype=np.complex128)
    rho = np.array(0.9+0j, dtype=np.complex128)
    epsilon = np.array(1e-6+0j, dtype=np.complex128)
    grad = np.array([0.5+0.5j, 0.6+0.6j, 0.7+0.7j], dtype=np.complex128)
    use_locking = False
    name = "adadelta_10"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
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
    
    print("Valid")

if 'tf.raw_ops.ApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdadelta'.")

check_valid('tf.raw_ops.ApplyAdadelta', generated_inputs['tf.raw_ops.ApplyAdadelta'], lib="tf", suffix=0)
