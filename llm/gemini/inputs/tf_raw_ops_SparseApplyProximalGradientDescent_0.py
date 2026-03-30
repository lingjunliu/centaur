
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyProximalGradientDescent_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([0.5, 0.0, 1.0, 0.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.1, dtype=np.float32)
    grad = np.array([0.5, 0.0, 1.0, 0.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int64)
    use_locking = True
    name = "test"

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1, 2, 3, 4], dtype=np.int32)
    alpha = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    grad = np.array([1, 0, 1, 0], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([0.5, 0.0, -1.0, 0.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    alpha = np.array(2.0, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([0.5, 0.0, 1.0, 0.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.5, dtype=np.float32)
    l2 = np.array(0.5, dtype=np.float32)
    grad = np.array([0.5, 0.0, 1.0, 0.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    alpha = np.array(0.1, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([0.5, 0.0, 1.0, 0.0], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    var = np.array([1.0+1j, 2.0+2j, 3.0+3j, 4.0+4j], dtype=np.complex64)
    alpha = np.array(0.1+0j, dtype=np.complex64)
    l1 = np.array(0.0+0j, dtype=np.complex64)
    l2 = np.array(0.0+0j, dtype=np.complex64)
    grad = np.array([0.5+0.5j, 0.0+0j, 1.0+1j, 0.0+0j], dtype=np.complex64)
    indices = np.array([0, 2], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "var": var,
        "alpha": alpha,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))



    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyProximalGradientDescent"] = tf_raw_ops_SparseApplyProximalGradientDescent_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyProximalGradientDescent' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyProximalGradientDescent'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseApplyProximalGradientDescent', generated_inputs['tf.raw_ops.SparseApplyProximalGradientDescent'], lib="tf", suffix=0)
