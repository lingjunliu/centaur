
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_RefSwitch_inputs():
    list_of_inputs = []

    with tf.compat.v1.Session() as sess:
        # Input 1
        data = tf.compat.v1.get_variable("data1", initializer=np.array(1, dtype=np.int32), use_resource=True)
        pred = tf.constant(True, dtype=tf.bool)
        name = "switch_op_1"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 2
        data = tf.compat.v1.get_variable("data2", initializer=np.array([1, 2, 3], dtype=np.float32), use_resource=True)
        pred = tf.constant(False, dtype=tf.bool)
        name = "switch_op_2"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 3
        data = tf.compat.v1.get_variable("data3", initializer=np.array([[1, 2], [3, 4]], dtype=np.int64), use_resource=True)
        pred = tf.constant(True, dtype=tf.bool)
        name = "switch_op_3"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 4
        data = tf.compat.v1.get_variable("data4", initializer=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64), use_resource=True)
        pred = tf.constant(False, dtype=tf.bool)
        name = "switch_op_4"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 5
        data = tf.compat.v1.get_variable("data5", initializer=np.array(-1, dtype=np.int32), use_resource=True)
        pred = tf.constant(True, dtype=tf.bool)
        name = "switch_op_5"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 6
        data = tf.compat.v1.get_variable("data6", initializer=np.array([-1.0, -2.0, -3.0], dtype=np.float32), use_resource=True)
        pred = tf.constant(False, dtype=tf.bool)
        name = "switch_op_6"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 7
        data = tf.compat.v1.get_variable("data7", initializer=np.array([[1, -2], [-3, 4]], dtype=np.int64), use_resource=True)
        pred = tf.constant(True, dtype=tf.bool)
        name = "switch_op_7"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 8
        data = tf.compat.v1.get_variable("data8", initializer=np.array([[[1, -2], [-3, 4]], [[5, -6], [-7, 8]]], dtype=np.float64), use_resource=True)
        pred = tf.constant(False, dtype=tf.bool)
        name = "switch_op_8"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 9
        data = tf.compat.v1.get_variable("data9", initializer=np.array(0, dtype=np.int32), use_resource=True)
        pred = tf.constant(True, dtype=tf.bool)
        name = "switch_op_9"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 10
        data = tf.compat.v1.get_variable("data10", initializer=np.array([1.0, 0.0, -1.0], dtype=np.float32), use_resource=True)
        pred = tf.constant(False, dtype=tf.bool)
        name = "switch_op_10"

        input_dict = {
            "data": data,
            "pred": pred,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RefSwitch"] = tf_raw_ops_RefSwitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RefSwitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefSwitch'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RefSwitch', generated_inputs['tf.raw_ops.RefSwitch'], lib="tf", suffix=0)
