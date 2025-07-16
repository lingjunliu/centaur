
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ApplyProximalAdagrad_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    use_locking = False
    input_dict = {"var": tf.Variable(var).numpy(), "accum": tf.Variable(accum).numpy(), "lr": tf.constant(lr).numpy(), "l1": tf.constant(l1).numpy(), "l2": tf.constant(l2).numpy(), "grad": tf.constant(grad).numpy(), "use_locking": use_locking, "name": "apply_proximal_adagrad_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    accum = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    lr = np.array(0.1, dtype=np.float64)
    l1 = np.array(0.01, dtype=np.float64)
    l2 = np.array(0.02, dtype=np.float64)
    grad = np.array([-0.5, -0.6, -0.7], dtype=np.float64)
    use_locking = True
    input_dict = {"var": tf.Variable(var).numpy(), "accum": tf.Variable(accum).numpy(), "lr": tf.constant(lr).numpy(), "l1": tf.constant(l1).numpy(), "l2": tf.constant(l2).numpy(), "grad": tf.constant(grad).numpy(), "use_locking": use_locking, "name": "apply_proximal_adagrad_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, positive values
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 2, 3], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(1, dtype=np.int32)
    l2 = np.array(1, dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    use_locking = False
    input_dict = {"var": tf.Variable(var, dtype=tf.int32).numpy(), "accum": tf.Variable(accum, dtype=tf.int32).numpy(), "lr": tf.constant(lr, dtype=tf.int32).numpy(), "l1": tf.constant(l1, dtype=tf.int32).numpy(), "l2": tf.constant(l2, dtype=tf.int32).numpy(), "grad": tf.constant(grad, dtype=tf.int32).numpy(), "use_locking": use_locking, "name": "apply_proximal_adagrad_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, negative values
    var = np.array([-1, -2, -3], dtype=np.int64)
    accum = np.array([1, 2, 3], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    l1 = np.array(1, dtype=np.int64)
    l2 = np.array(1, dtype=np.int64)
    grad = np.array([-1, -1, -1], dtype=np.int64)
    use_locking = True
    input_dict = {"var": tf.Variable(var, dtype=tf.int64).numpy(), "accum": tf.Variable(accum, dtype=tf.int64).numpy(), "lr": tf.constant(lr, dtype=tf.int64).numpy(), "l1": tf.constant(l1, dtype=tf.int64).numpy(), "l2": tf.constant(l2, dtype=tf.int64).numpy(), "grad": tf.constant(grad, dtype=tf.int64).numpy(), "use_locking": use_locking, "name": "apply_proximal_adagrad_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes (1D) float32
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    grad = np.array([0.5], dtype=np.float32)
    use_locking = False
    input_dict = {"var": tf.Variable(var).numpy(), "accum": tf.Variable(accum).numpy(), "lr": tf.constant(lr).numpy(), "l1": tf.constant(l1).numpy(), "l2": tf.constant(l2).numpy(), "grad": tf.constant(grad).numpy(), "use_locking": use_locking, "name": "apply_proximal_adagrad_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger values for accum
    var = np.array([1.0, 2.0], dtype=np.float32)
    accum = np.array([100.0, 200.0], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.001, dtype=np.float32)
    l2 = np.array(0.002, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    use_locking = False
    input_dict = {"var": tf.Variable(var).numpy(), "accum": tf.Variable(accum).numpy(), "lr": tf.constant(lr).numpy(), "l1": tf.constant(l1).numpy(), "l2": tf.constant(l2).numpy(), "grad": tf.constant(grad).numpy(), "use_locking": use_locking, "name": "apply_proximal_adagrad_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero values for lr, l1, l2
    var = np.array([1.0, 2.0], dtype=np.float32)
    accum = np.array([0.1, 0.2], dtype=np.float32)
    lr = np.array(0.0, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    use_locking = False
    input_dict = {"var": tf.Variable(var).numpy(), "accum": tf.Variable(accum).numpy(), "lr": tf.constant(lr).numpy(), "l1": tf.constant(l1).numpy(), "l2": tf.constant(l2).numpy(), "grad": tf.constant(grad).numpy(), "use_locking": use_locking, "name": "apply_proximal_adagrad_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: half type
    var = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    lr = np.array(0.01, dtype=np.float16)
    l1 = np.array(0.001, dtype=np.float16)
    l2 = np.array(0.002, dtype=np.float16)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float16)
    use_locking = False
    input_dict = {"var": tf.Variable(var).numpy(), "accum": tf.Variable(accum).numpy(), "lr": tf.constant(lr).numpy(), "l1": tf.constant(l1).numpy(), "l2": tf.constant(l2).numpy(), "grad": tf.constant(grad).numpy(), "use_locking": use_locking, "name": "apply_proximal_adagrad_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyProximalAdagrad"] = tf_raw_ops_ApplyProximalAdagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyProximalAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyProximalAdagrad'.")

check_valid('tf.raw_ops.ApplyProximalAdagrad', generated_inputs['tf.raw_ops.ApplyProximalAdagrad'], lib="tf", suffix=0)
