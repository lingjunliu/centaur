
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

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
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_1"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    rho = np.array(0.9, dtype=np.float64)
    epsilon = np.array(1e-6, dtype=np.float64)
    grad = np.array([0.5, 0.6], dtype=np.float64)
    indices = np.array([0], dtype=np.int64)
    use_locking = True
    name = "sparse_apply_adadelta_2"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    accum = np.array([[1, 2], [3, 4]], dtype=np.int32)
    accum_update = np.array([[1, 2], [3, 4]], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    rho = np.array(1, dtype=np.int32)
    epsilon = np.array(1, dtype=np.int32)
    grad = np.array([1, 2], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_3"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]], dtype=np.float32)
    lr = np.array(0.005, dtype=np.float32)
    rho = np.array(0.95, dtype=np.float32)
    epsilon = np.array(1e-8, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    indices = np.array([1], dtype=np.int32)
    use_locking = True
    name = "sparse_apply_adadelta_4"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    accum_update = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    lr = np.array(0.02, dtype=np.float32)
    rho = np.array(0.8, dtype=np.float32)
    epsilon = np.array(1e-5, dtype=np.float32)
    grad = np.array(0.2, dtype=np.float32)
    indices = np.array([2], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_5"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(-0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_6"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    accum = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    accum_update = np.array([[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_7"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(-0.9, dtype=np.float32)
    epsilon = np.array(1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_8"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    accum_update = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    rho = np.array(0.9, dtype=np.float32)
    epsilon = np.array(-1e-6, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "sparse_apply_adadelta_9"
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
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
    input_dict = {"var": var, "accum": accum, "accum_update": accum_update, "lr": lr, "rho": rho, "epsilon": epsilon, "grad": grad, "indices": indices, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseApplyAdadelta"] = tf_raw_ops_SparseApplyAdadelta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseApplyAdadelta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyAdadelta'.")

check_valid('tf.raw_ops.SparseApplyAdadelta', generated_inputs['tf.raw_ops.SparseApplyAdadelta'], lib="tf", suffix=0)
