
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logsoftmax_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"logits": tf.constant(logits), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, with negative values
    logits = np.array([[-1.0, 0.0], [-2.0, -3.0]], dtype=np.float64)
    input_dict = {"logits": tf.constant(logits), "name": "log_softmax_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half (float16), different values
    logits = np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float16)
    input_dict = {"logits": tf.constant(logits), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16, all zeros
    logits = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=tf.bfloat16.as_numpy_dtype)
    input_dict = {"logits": tf.constant(logits), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, larger values
    logits = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    input_dict = {"logits": tf.constant(logits), "name": "another_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, all same values
    logits = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float64)
    input_dict = {"logits": tf.constant(logits), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: half, small values
    logits = np.array([[0.001, 0.002], [0.003, 0.004]], dtype=np.float16)
    input_dict = {"logits": tf.constant(logits), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16, negative and positive
    logits = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=tf.bfloat16.as_numpy_dtype)
    input_dict = {"logits": tf.constant(logits), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, more diverse values
    logits = np.array([[-5.2, 2.7], [0.1, -10.5]], dtype=np.float32)
    input_dict = {"logits": tf.constant(logits), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, large negative value
    logits = np.array([[1.0, -1000.0], [2.0, -2000.0]], dtype=np.float64)
    input_dict = {"logits": tf.constant(logits), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LogSoftmax"] = tf_raw_ops_logsoftmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LogSoftmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LogSoftmax'.")

check_valid('tf.raw_ops.LogSoftmax', generated_inputs['tf.raw_ops.LogSoftmax'], lib="tf", suffix=0)
