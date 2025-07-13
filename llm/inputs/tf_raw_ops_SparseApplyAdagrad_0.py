
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyAdagrad_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.5, 0.6]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    update_slots = True
    name = "sparse_apply_adagrad_1"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = True
    update_slots = False
    name = "sparse_apply_adagrad_2"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    grad = np.array([[0.5, 0.6]], dtype=np.float64)
    indices = np.array([0], dtype=np.int64)
    use_locking = False
    update_slots = True
    name = "sparse_apply_adagrad_3"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices = np.array([1, 0], dtype=np.int32)
    use_locking = False
    update_slots = True
    name = "sparse_apply_adagrad_4"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.5], dtype=np.float32)
    indices = np.array([1], dtype=np.int32)
    use_locking = False
    update_slots = True
    name = "sparse_apply_adagrad_5"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    use_locking = True
    update_slots = True
    name = "sparse_apply_adagrad_6"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    update_slots = True
    name = "sparse_apply_adagrad_7"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    use_locking = True
    update_slots = False
    name = "sparse_apply_adagrad_8"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.5, dtype=np.float32)
    grad = np.array([[0.1, 0.2]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    update_slots = True
    name = "sparse_apply_adagrad_9"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = False
    update_slots = True
    name = ""

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "update_slots": update_slots,
        "name": name
    }
    for key in ["var", "accum", "lr", "grad", "indices"]:
        input_dict[key] = tf.convert_to_tensor(input_dict[key])
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyAdagrad"] = tf_raw_ops_SparseApplyAdagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseApplyAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyAdagrad'.")

check_valid('tf.raw_ops.SparseApplyAdagrad', generated_inputs['tf.raw_ops.SparseApplyAdagrad'], lib="tf", suffix=0)
