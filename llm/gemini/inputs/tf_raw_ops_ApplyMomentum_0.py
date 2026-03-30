
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_apply_momentum_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_1"

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var1", initializer=var)
        a = tf.compat.v1.get_variable("accum1", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())

        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    lr = np.array(0.005, dtype=np.float64)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    momentum = np.array(0.8, dtype=np.float64)
    use_locking = True
    use_nesterov = True
    name = "momentum_update_2"

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var2", initializer=var)
        a = tf.compat.v1.get_variable("accum2", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())

        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([0, 0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    momentum = np.array(1, dtype=np.int32)
    use_locking = False
    use_nesterov = True
    name = "momentum_update_3"
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var3", initializer=var)
        a = tf.compat.v1.get_variable("accum3", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())


        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    accum = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([-0.4, -0.5, -0.6], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = True
    use_nesterov = False
    name = "momentum_update_4"

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var4", initializer=var)
        a = tf.compat.v1.get_variable("accum4", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())

        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.4], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_5"

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var5", initializer=var)
        a = tf.compat.v1.get_variable("accum5", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())

        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([1, 2, 3], dtype=np.int64)
    accum = np.array([0, 0, 0], dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    grad = np.array([1, 1, 1], dtype=np.int64)
    momentum = np.array(1, dtype=np.int64)
    use_locking = False
    use_nesterov = True
    name = "momentum_update_6"
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var6", initializer=var)
        a = tf.compat.v1.get_variable("accum6", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())

        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.1, dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    momentum = np.array(0.5, dtype=np.float32)
    use_locking = True
    use_nesterov = True
    name = "momentum_update_7"

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var7", initializer=var)
        a = tf.compat.v1.get_variable("accum7", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())


        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    grad = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    momentum = np.array(0.9, dtype=np.float64)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_8"

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var8", initializer=var)
        a = tf.compat.v1.get_variable("accum8", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())

        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([0, 0, 0], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    momentum = np.array(1, dtype=np.int32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_9"

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var9", initializer=var)
        a = tf.compat.v1.get_variable("accum9", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())

        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = True
    use_nesterov = True
    name = "momentum_update_10"

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        v = tf.compat.v1.get_variable("var10", initializer=var)
        a = tf.compat.v1.get_variable("accum10", initializer=accum)
        sess.run(tf.compat.v1.initialize_all_variables())

        input_dict = {
            "var": v,
            "accum": a,
            "lr": lr,
            "grad": grad,
            "momentum": momentum,
            "use_locking": use_locking,
            "use_nesterov": use_nesterov,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyMomentum"] = tf_raw_ops_apply_momentum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyMomentum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyMomentum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyMomentum', generated_inputs['tf.raw_ops.ApplyMomentum'], lib="tf", suffix=0)
