
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

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
    name = "test1"

    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var1", initializer=var)
        a = tf.compat.v1.get_variable("accum1", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)
    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    l1 = np.array(0.1, dtype=np.float64)
    l2 = np.array(0.01, dtype=np.float64)
    grad = np.array([0.2], dtype=np.float64)
    indices = np.array([1], dtype=np.int64)
    use_locking = True
    name = "test2"
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var2", initializer=var)
        a = tf.compat.v1.get_variable("accum2", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)

    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    accum = np.array([[5, 6], [7, 8]], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    grad = np.array([[1, 1]], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "test3"
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var3", initializer=var)
        a = tf.compat.v1.get_variable("accum3", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)

    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    lr = np.array(0.5, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([0.2], dtype=np.float32)
    indices = np.array([0], dtype=np.int64)
    use_locking = True
    name = "test4"
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var4", initializer=var)
        a = tf.compat.v1.get_variable("accum4", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)

    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    accum = np.array([[0.1, -0.2], [-0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([[0.5, 0.6]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    use_locking = False
    name = "test5"
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var5", initializer=var)
        a = tf.compat.v1.get_variable("accum5", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)

    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    l1 = np.array(0.1, dtype=np.float64)
    l2 = np.array(0.01, dtype=np.float64)
    grad = np.array([0.2, 0.3], dtype=np.float64)
    indices = np.array([1, 2], dtype=np.int64)
    use_locking = True
    name = "test6"
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var6", initializer=var)
        a = tf.compat.v1.get_variable("accum6", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)


    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    accum = np.array([[5, 6], [7, 8]], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    grad = np.array([[1, 1], [2,2]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    use_locking = False
    name = "test7"
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var7", initializer=var)
        a = tf.compat.v1.get_variable("accum7", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)


    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    lr = np.array(0.5, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([0.2], dtype=np.float32)
    indices = np.array([0], dtype=np.int64)
    use_locking = True
    name = "test8"
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var8", initializer=var)
        a = tf.compat.v1.get_variable("accum8", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)

    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    accum = np.array([[0.1, -0.2], [-0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    grad = np.array([[0.5, 0.6],[0.1, 0.2]], dtype=np.float32)
    indices = np.array([0,1], dtype=np.int32)
    use_locking = False
    name = "test9"
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var9", initializer=var)
        a = tf.compat.v1.get_variable("accum9", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)

    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "grad": grad,
        "indices": indices,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    lr = np.array(0.001, dtype=np.float64)
    l1 = np.array(0.1, dtype=np.float64)
    l2 = np.array(0.01, dtype=np.float64)
    grad = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    indices = np.array([1, 2, 3], dtype=np.int64)
    use_locking = True
    name = "test10"
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var10", initializer=var)
        a = tf.compat.v1.get_variable("accum10", initializer=accum)
        sess.run(tf.compat.v1.global_variables_initializer())
        v_val = sess.run(v)
        a_val = sess.run(a)

    input_dict = {
        "var": v,
        "accum": a,
        "lr": lr,
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
generated_inputs["tf.raw_ops.SparseApplyProximalAdagrad"] = tf_raw_ops_SparseApplyProximalAdagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyProximalAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyProximalAdagrad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseApplyProximalAdagrad', generated_inputs['tf.raw_ops.SparseApplyProximalAdagrad'], lib="tf", suffix=0)
