
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ApplyAdagrad_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float32
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    use_locking = False
    update_slots = True
    name = "adagrad_1"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values, use_locking=True
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    accum = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    grad = np.array([-0.5, -0.6, -0.7], dtype=np.float64)
    use_locking = True
    update_slots = False
    name = "adagrad_2"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, update_slots=False
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 2, 3], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    use_locking = False
    update_slots = False
    name = "adagrad_3"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    use_locking = False
    update_slots = True
    name = "adagrad_4"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64
    var = np.array([1, 2, 3], dtype=np.int64)
    accum = np.array([1, 2, 3], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    grad = np.array([1, 1, 1], dtype=np.int64)
    use_locking = True
    update_slots = True
    name = "adagrad_5"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64
    var = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    accum = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex64)
    lr = np.array(0.01+0.01j, dtype=np.complex64)
    grad = np.array([0.5+0.5j, 0.6+0.6j, 0.7+0.7j], dtype=np.complex64)
    use_locking = False
    update_slots = True
    name = "adagrad_6"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Two dimensional complex128
    var = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    accum = np.array([[0.1+0.1j, 0.2+0.2j], [0.3+0.3j, 0.4+0.4j]], dtype=np.complex128)
    lr = np.array(0.01+0.01j, dtype=np.complex128)
    grad = np.array([[0.5+0.5j, 0.6+0.6j], [0.7+0.7j, 0.8+0.8j]], dtype=np.complex128)
    use_locking = True
    update_slots = False
    name = "adagrad_7"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16
    var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float16)
    use_locking = False
    update_slots = True
    name = "adagrad_8"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32,
    var = np.array([1, 2, 3], dtype=np.uint32)
    accum = np.array([1, 2, 3], dtype=np.uint32)
    lr = np.array(1, dtype=np.uint32)
    grad = np.array([1, 1, 1], dtype=np.uint32)
    use_locking = True
    update_slots = True
    name = "adagrad_9"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint64,
    var = np.array([1, 2, 3], dtype=np.uint64)
    accum = np.array([1, 2, 3], dtype=np.uint64)
    lr = np.array(1, dtype=np.uint64)
    grad = np.array([1, 1, 1], dtype=np.uint64)
    use_locking = False
    update_slots = False
    name = "adagrad_10"
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "use_locking": use_locking, "update_slots": update_slots, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyAdagrad"] = tf_raw_ops_ApplyAdagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagrad'.")

check_valid('tf.raw_ops.ApplyAdagrad', generated_inputs['tf.raw_ops.ApplyAdagrad'], lib="tf", suffix=0)
