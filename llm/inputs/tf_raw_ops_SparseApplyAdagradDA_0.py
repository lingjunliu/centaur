
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyAdagradDA_inputs():
    list_of_inputs = []

    # Input 1: float32
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_float32_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    grad = np.array([0.5, 0.6], dtype=np.float64)
    indices = np.array([1], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float64)
    l1 = np.array(0.0, dtype=np.float64)
    l2 = np.array(0.0, dtype=np.float64)
    global_step = np.array(1, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_float64_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    gradient_accumulator = np.array([[1, 2], [3, 4]], dtype=np.int32)
    gradient_squared_accumulator = np.array([[1, 2], [3, 4]], dtype=np.int32)
    grad = np.array([5, 6], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_int32_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8
    var = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    gradient_accumulator = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    gradient_squared_accumulator = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    grad = np.array([5, 6], dtype=np.uint8)
    indices = np.array([1], dtype=np.int64)
    lr = np.array(1, dtype=np.uint8)
    l1 = np.array(0, dtype=np.uint8)
    l2 = np.array(0, dtype=np.uint8)
    global_step = np.array(1, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_uint8_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int16
    var = np.array([[1, 2], [3, 4]], dtype=np.int16)
    gradient_accumulator = np.array([[1, 2], [3, 4]], dtype=np.int16)
    gradient_squared_accumulator = np.array([[1, 2], [3, 4]], dtype=np.int16)
    grad = np.array([5, 6], dtype=np.int16)
    indices = np.array([0,1], dtype=np.int32)
    lr = np.array(1, dtype=np.int16)
    l1 = np.array(0, dtype=np.int16)
    l2 = np.array(0, dtype=np.int16)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_int16_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int8
    var = np.array([[1, 2], [3, 4]], dtype=np.int8)
    gradient_accumulator = np.array([[1, 2], [3, 4]], dtype=np.int8)
    gradient_squared_accumulator = np.array([[1, 2], [3, 4]], dtype=np.int8)
    grad = np.array([5, 6], dtype=np.int8)
    indices = np.array([1], dtype=np.int64)
    lr = np.array(1, dtype=np.int8)
    l1 = np.array(0, dtype=np.int8)
    l2 = np.array(0, dtype=np.int8)
    global_step = np.array(1, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_int8_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: complex64
    var = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    gradient_accumulator = np.array([[0.1+0.1j, 0.2+0.2j], [0.3+0.3j, 0.4+0.4j]], dtype=np.complex64)
    gradient_squared_accumulator = np.array([[0.01+0.01j, 0.02+0.02j], [0.03+0.03j, 0.04+0.04j]], dtype=np.complex64)
    grad = np.array([0.5+0.5j, 0.6+0.6j], dtype=np.complex64)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01+0.01j, dtype=np.complex64)
    l1 = np.array(0.0+0.0j, dtype=np.complex64)
    l2 = np.array(0.0+0.0j, dtype=np.complex64)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_complex64_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128
    var = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    gradient_accumulator = np.array([[0.1+0.1j, 0.2+0.2j], [0.3+0.3j, 0.4+0.4j]], dtype=np.complex128)
    gradient_squared_accumulator = np.array([[0.01+0.01j, 0.02+0.02j], [0.03+0.03j, 0.04+0.04j]], dtype=np.complex128)
    grad = np.array([0.5+0.5j, 0.6+0.6j], dtype=np.complex128)
    indices = np.array([1], dtype=np.int64)
    lr = np.array(0.01+0.01j, dtype=np.complex128)
    l1 = np.array(0.0+0.0j, dtype=np.complex128)
    l2 = np.array(0.0+0.0j, dtype=np.complex128)
    global_step = np.array(1, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_complex128_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: half
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float16)
    grad = np.array([0.5, 0.6], dtype=np.float16)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float16)
    l1 = np.array(0.0, dtype=np.float16)
    l2 = np.array(0.0, dtype=np.float16)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_half_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64
    var = np.array([[1, 2], [3, 4]], dtype=np.int64)
    gradient_accumulator = np.array([[1, 2], [3, 4]], dtype=np.int64)
    gradient_squared_accumulator = np.array([[1, 2], [3, 4]], dtype=np.int64)
    grad = np.array([5, 6], dtype=np.int64)
    indices = np.array([1], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    l1 = np.array(0, dtype=np.int64)
    l2 = np.array(0, dtype=np.int64)
    global_step = np.array(1, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_int64_1"

    input_dict = {
        "var": var,
        "gradient_accumulator": gradient_accumulator,
        "gradient_squared_accumulator": gradient_squared_accumulator,
        "grad": grad,
        "indices": indices,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "global_step": global_step,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyAdagradDA"] = tf_raw_ops_SparseApplyAdagradDA_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseApplyAdagradDA' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyAdagradDA'.")

check_valid('tf.raw_ops.SparseApplyAdagradDA', generated_inputs['tf.raw_ops.SparseApplyAdagradDA'], lib="tf", suffix=0)
