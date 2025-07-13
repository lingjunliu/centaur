
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyProximalAdagrad_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([[0.5, 0.6]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_proximal_adagrad_1"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.float32),
        "accum": tf.convert_to_tensor(accum, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = True
    name = "sparse_apply_proximal_adagrad_2"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.float32),
        "accum": tf.convert_to_tensor(accum, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    accum = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    lr = np.array(0.005, dtype=np.float64)
    l1 = np.array(0.05, dtype=np.float64)
    l2 = np.array(0.005, dtype=np.float64)
    grad = np.array([[0.2, 0.3, 0.4]], dtype=np.float64)
    indices = np.array([1], dtype=np.int64)
    use_locking = False
    name = "sparse_apply_proximal_adagrad_3"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.float64),
        "accum": tf.convert_to_tensor(accum, dtype=tf.float64),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float64),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float64),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float64),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float64),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    var = np.array([1, 2, 3, 4], dtype=np.int32)
    accum = np.array([0, 0, 0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    grad = np.array([1, 1], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_proximal_adagrad_4"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.int32),
        "accum": tf.convert_to_tensor(accum, dtype=tf.int32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.int32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.int32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    accum = np.array([[0.1+0.1j, 0.2+0.2j], [0.3+0.3j, 0.4+0.4j]], dtype=np.complex64)
    lr = np.array(0.01+0.01j, dtype=np.complex64)
    l1 = np.array(0.0+0.0j, dtype=np.complex64)
    l2 = np.array(0.0+0.0j, dtype=np.complex64)
    grad = np.array([[0.5+0.5j, 0.6+0.6j]], dtype=np.complex64)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_proximal_adagrad_5"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.complex64),
        "accum": tf.convert_to_tensor(accum, dtype=tf.complex64),
        "lr": tf.convert_to_tensor(lr, dtype=tf.complex64),
        "l1": tf.convert_to_tensor(l1, dtype=tf.complex64),
        "l2": tf.convert_to_tensor(l2, dtype=tf.complex64),
        "grad": tf.convert_to_tensor(grad, dtype=tf.complex64),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([1, 2, 3, 4], dtype=np.int64)
    accum = np.array([0, 0, 0, 0], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    l1 = np.array(0, dtype=np.int64)
    l2 = np.array(0, dtype=np.int64)
    grad = np.array([1, 1], dtype=np.int64)
    indices = np.array([0, 2], dtype=np.int64)
    use_locking = False
    name = "sparse_apply_proximal_adagrad_6"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.int64),
        "accum": tf.convert_to_tensor(accum, dtype=tf.int64),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int64),
        "l1": tf.convert_to_tensor(l1, dtype=tf.int64),
        "l2": tf.convert_to_tensor(l2, dtype=tf.int64),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int64),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices = np.array([0,1], dtype=np.int64)
    use_locking = False
    name = "sparse_apply_proximal_adagrad_7"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.float32),
        "accum": tf.convert_to_tensor(accum, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = np.array([1, 2, 3], dtype=np.uint8)
    accum = np.array([0, 0, 0], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    l1 = np.array(0, dtype=np.uint8)
    l2 = np.array(0, dtype=np.uint8)
    grad = np.array([1, 1], dtype=np.uint8)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_proximal_adagrad_8"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.uint8),
        "accum": tf.convert_to_tensor(accum, dtype=tf.uint8),
        "lr": tf.convert_to_tensor(lr, dtype=tf.uint8),
        "l1": tf.convert_to_tensor(l1, dtype=tf.uint8),
        "l2": tf.convert_to_tensor(l2, dtype=tf.uint8),
        "grad": tf.convert_to_tensor(grad, dtype=tf.uint8),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    l1 = np.array(0.1, dtype=np.float16)
    l2 = np.array(0.01, dtype=np.float16)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = True
    name = "sparse_apply_proximal_adagrad_9"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.float16),
        "accum": tf.convert_to_tensor(accum, dtype=tf.float16),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float16),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float16),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float16),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float16),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int32),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.005, dtype=np.float32)
    l1 = np.array(0.05, dtype=np.float32)
    l2 = np.array(0.005, dtype=np.float32)
    grad = np.array([0.2, 0.3], dtype=np.float32)
    indices = np.array([1, 2], dtype=np.int64)
    use_locking = False
    name = "sparse_apply_proximal_adagrad_10"

    input_dict = {
        "var": tf.convert_to_tensor(var, dtype=tf.float32),
        "accum": tf.convert_to_tensor(accum, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "indices": tf.convert_to_tensor(indices, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyProximalAdagrad"] = tf_raw_ops_SparseApplyProximalAdagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseApplyProximalAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyProximalAdagrad'.")

check_valid('tf.raw_ops.SparseApplyProximalAdagrad', generated_inputs['tf.raw_ops.SparseApplyProximalAdagrad'], lib="tf", suffix=0)
