
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_apply_adagrad_inputs():
    list_of_inputs = []

    # Input 1
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var1", shape=(3,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)))
        accum = tf.compat.v1.get_variable("accum1", shape=(3,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([0.1, 0.2, 0.3], dtype=np.float32)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(0.01, dtype=np.float32)
        grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        use_locking = False
        update_slots = True
        name = "adagrad_1"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var2", shape=(2,2), dtype=tf.float64, initializer=tf.compat.v1.initializers.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)))
        accum = tf.compat.v1.get_variable("accum2", shape=(2,2), dtype=tf.float64, initializer=tf.compat.v1.initializers.constant(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(0.005, dtype=np.float64)
        grad = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float64)
        use_locking = True
        update_slots = False
        name = "adagrad_2"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var3", shape=(3,), dtype=tf.int32, initializer=tf.compat.v1.initializers.constant(np.array([1, 2, 3], dtype=np.int32)))
        accum = tf.compat.v1.get_variable("accum3", shape=(3,), dtype=tf.int32, initializer=tf.compat.v1.initializers.constant(np.array([1, 2, 3], dtype=np.int32)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(1, dtype=np.int32)
        grad = np.array([1, 1, 1], dtype=np.int32)
        use_locking = False
        update_slots = True
        name = "adagrad_3"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var4", shape=(2,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([-1.0, -2.0], dtype=np.float32)))
        accum = tf.compat.v1.get_variable("accum4", shape=(2,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([0.5, 0.5], dtype=np.float32)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(0.1, dtype=np.float32)
        grad = np.array([0.2, 0.3], dtype=np.float32)
        use_locking = True
        update_slots = False
        name = "adagrad_4"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var5", shape=(1,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([1.0], dtype=np.float32)))
        accum = tf.compat.v1.get_variable("accum5", shape=(1,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([0.1], dtype=np.float32)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(0.01, dtype=np.float32)
        grad = np.array([0.5], dtype=np.float32)
        use_locking = False
        update_slots = True
        name = "adagrad_5"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var6", shape=(2,), dtype=tf.int64, initializer=tf.compat.v1.initializers.constant(np.array([1, 2], dtype=np.int64)))
        accum = tf.compat.v1.get_variable("accum6", shape=(2,), dtype=tf.int64, initializer=tf.compat.v1.initializers.constant(np.array([1, 2], dtype=np.int64)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(1, dtype=np.int64)
        grad = np.array([1, 1], dtype=np.int64)
        use_locking = False
        update_slots = True
        name = "adagrad_6"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var7", shape=(4,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)))
        accum = tf.compat.v1.get_variable("accum7", shape=(4,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(0.001, dtype=np.float32)
        grad = np.array([0.05, 0.06, 0.07, 0.08], dtype=np.float32)
        use_locking = True
        update_slots = False
        name = "adagrad_7"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var8", shape=(3,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([0.5, 0.6, 0.7], dtype=np.float32)))
        accum = tf.compat.v1.get_variable("accum8", shape=(3,), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([0.05, 0.06, 0.07], dtype=np.float32)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(0.0001, dtype=np.float32)
        grad = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        use_locking = False
        update_slots = True
        name = "adagrad_8"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var9", shape=(2,2), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)))
        accum = tf.compat.v1.get_variable("accum9", shape=(2,2), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(0.00001, dtype=np.float32)
        grad = np.array([[0.001, 0.002], [0.003, 0.004]], dtype=np.float32)
        use_locking = True
        update_slots = False
        name = "adagrad_9"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    with tf.compat.v1.Session() as sess:
        var = tf.compat.v1.get_variable("var10", shape=(2,2), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32)))
        accum = tf.compat.v1.get_variable("accum10", shape=(2,2), dtype=tf.float32, initializer=tf.compat.v1.initializers.constant(np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)))
        sess.run(tf.compat.v1.global_variables_initializer())
        lr = np.array(0.0001, dtype=np.float32)
        grad = np.array([[0.001, 0.002], [0.003, 0.004]], dtype=np.float32)
        use_locking = False
        update_slots = True
        name = "adagrad_10"

        input_dict = {
            "var": var,
            "accum": accum,
            "lr": lr,
            "grad": grad,
            "use_locking": use_locking,
            "update_slots": update_slots,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyAdagrad"] = tf_raw_ops_apply_adagrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdagrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagrad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyAdagrad', generated_inputs['tf.raw_ops.ApplyAdagrad'], lib="tf", suffix=0)
