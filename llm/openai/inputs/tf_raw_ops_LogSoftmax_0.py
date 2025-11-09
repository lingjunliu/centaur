
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logsoftmax_inputs():
    list_of_inputs = []

    logits = np.array([[1.0, 2.0, 3.0], [0.5, -1.5, 2.5]], dtype=np.float32)
    name = "logsoftmax_input_1"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[-1.0, -2.0, -3.0, -4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float64)
    name = "logsoftmax_input_2"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[100.0, -100.0], [0.0, 0.0], [1.0, -1.0]], dtype=np.float16)
    name = "logsoftmax_input_3"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.linspace(-5, 5, num=20, dtype=np.float32).reshape(1, -1)
    name = "logsoftmax_input_4"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.empty((0, 5), dtype=np.float32)
    name = "logsoftmax_input_5"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[3.14], [-2.71], [0.0]], dtype=np.float64)
    name = "logsoftmax_input_6"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[np.inf, 0.0, -np.inf], [np.nan, 1.0, -1.0]], dtype=np.float64)
    name = "logsoftmax_input_7"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -999.5, -1001.0]], dtype=np.float32)
    name = "logsoftmax_input_8"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    rng = np.random.default_rng(42)
    logits = rng.normal(loc=0.0, scale=5.0, size=(4, 4)).astype(np.float16)
    name = "logsoftmax_input_9"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[-0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    name = "logsoftmax_input_10"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                       [-0.5, -0.4, -0.3, -0.2, -0.1]], dtype=np.float64)
    name = "logsoftmax_input_11"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = rng.uniform(low=-10.0, high=10.0, size=(2, 50)).astype(np.float32)
    name = "logsoftmax_input_12"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    return list_of_inputs

generated_inputs["tf.raw_ops.LogSoftmax"] = tf_raw_ops_logsoftmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LogSoftmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LogSoftmax'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.LogSoftmax', generated_inputs['tf.raw_ops.LogSoftmax'], lib="tf", suffix=0)
