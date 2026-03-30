
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_SparseApplyAdadelta_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_1"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = True
    name = "sparse_apply_adadelta_2"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - different type, int32
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    accum = np.array([[0, 1], [2, 3]], dtype=np.int32)
    accum_update = np.array([[0, 0], [1, 1]], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(0, dtype=np.int32)
    epsilon = np.array(0, dtype=np.int32)
    grad = np.array([1, 1], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_3"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4 - int64 indices
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    use_locking = True
    name = "sparse_apply_adadelta_4"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - different scalar values
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    rho = np.array(0.5, dtype=np.float32)
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_5"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - single index and grad.
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.6], dtype=np.float32)
    indices = np.array([1], dtype=np.int32)
    use_locking = True
    name = "sparse_apply_adadelta_6"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Float64 type and grad
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    epsilon = np.array(1e-6, dtype=np.float64)
    grad = np.array([0.5], dtype=np.float64)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_7"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - lr=0, rho=0, epsilon=0 and grad
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.0, dtype=np.float32)
    rho = np.array(0.0, dtype=np.float32)
    epsilon = np.array(0.0, dtype=np.float32)
    grad = np.array([0.5], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_8"

    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - multi-dimensional var. Removing because it seems to have caused shape mismatch issues
    # var = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    # accum = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    # accum_update = np.array([[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]]], dtype=np.float32)
    # lr = np.array(0.01, dtype=np.float32)
    # rho = np.array(0.9, dtype=np.float32)
    # epsilon = np.array(1e-6, dtype=np.float32)
    # grad = np.array([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    # indices = np.array([0], dtype=np.int32)
    # use_locking = False
    # name = "sparse_apply_adadelta_9"
    
    # input_dict = {
    #     "var": var,
    #     "accum": accum,
    #     "accum_update": accum_update,
    #     "lr": lr,
    #     "rho": rho,
    #     "epsilon": epsilon,
    #     "grad": grad,
    #     "indices": indices,
    #     "use_locking": use_locking,
    #     "name": name
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 10 - match shape to var
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0,1], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_10"
    
    input_dict = {
        "var": var,
        "accum": accum,
        "accum_update": accum_update,
        "lr": lr,
        "rho": rho,
        "epsilon": epsilon,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyAdadelta"] = tf_raw_ops_SparseApplyAdadelta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyAdadelta'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseApplyAdadelta', generated_inputs['tf.raw_ops.SparseApplyAdadelta'], lib="tf", suffix=0)
