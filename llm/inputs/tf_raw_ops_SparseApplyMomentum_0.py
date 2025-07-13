
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyMomentum_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    grad = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    momentum = np.array(0.5, dtype=np.float32)
    use_locking = True
    use_nesterov = True

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    accum = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    lr = np.array(0.2, dtype=np.float64)
    grad = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    indices = np.array([0, 2], dtype=np.int32)
    momentum = np.array(0.7, dtype=np.float64)
    use_locking = False
    use_nesterov = False

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    accum = np.array([[0, 0], [0, 0]], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    grad = np.array([[1, 1], [1, 1]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int64)
    momentum = np.array(1, dtype=np.int32)
    use_locking = True
    use_nesterov = True

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([1, 2, 3], dtype=np.int64)
    accum = np.array([0, 0, 0], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    grad = np.array([1, 1, 1], dtype=np.int64)
    indices = np.array([0, 2], dtype=np.int32)
    momentum = np.array(1, dtype=np.int64)
    use_locking = False
    use_nesterov = False

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    accum = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float16)
    lr = np.array(0.1, dtype=np.float16)
    grad = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    momentum = np.array(0.9, dtype=np.float16)
    use_locking = False
    use_nesterov = False

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: half
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    accum = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float16)
    lr = np.array(0.1, dtype=np.float16)
    grad = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    momentum = np.array(0.9, dtype=np.float16)
    use_locking = False
    use_nesterov = False

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8
    var = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    accum = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    grad = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    indices = np.array([0, 1], dtype=np.int32)
    momentum = np.array(1, dtype=np.uint8)
    use_locking = False
    use_nesterov = False

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: uint64
    var = np.array([1, 2, 3], dtype=np.uint64)
    accum = np.array([0, 0, 0], dtype=np.uint64)
    lr = np.array(1, dtype=np.uint64)
    grad = np.array([1, 1, 1], dtype=np.uint64)
    indices = np.array([0, 2], dtype=np.int64)
    momentum = np.array(1, dtype=np.uint64)
    use_locking = False
    use_nesterov = False

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex64
    var = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    accum = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex64)
    lr = np.array(0.1+0.1j, dtype=np.complex64)
    grad = np.array([0.5+0.5j, 0.5+0.5j, 0.5+0.5j], dtype=np.complex64)
    indices = np.array([0, 2], dtype=np.int32)
    momentum = np.array(0.9+0.9j, dtype=np.complex64)
    use_locking = False
    use_nesterov = False

    input_dict = {
        "var": tf.Variable(var).value(),
        "accum": tf.Variable(accum).value(),
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": "momentum_update_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyMomentum"] = tf_raw_ops_SparseApplyMomentum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseApplyMomentum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyMomentum'.")

check_valid('tf.raw_ops.SparseApplyMomentum', generated_inputs['tf.raw_ops.SparseApplyMomentum'], lib="tf", suffix=0)
