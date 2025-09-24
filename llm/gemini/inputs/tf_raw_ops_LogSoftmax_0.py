
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logsoftmax_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"logits": logits, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, negative values
    logits = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float64)
    input_dict = {"logits": logits, "name": "log_softmax_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, zero values
    logits = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    input_dict = {"logits": logits, "name": "log_softmax_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half (float16)
    logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {"logits": logits, "name": "log_softmax_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, larger batch size
    logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    input_dict = {"logits": logits, "name": "log_softmax_large_batch"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, larger number of classes
    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float64)
    input_dict = {"logits": logits, "name": "log_softmax_more_classes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, mixed positive and negative values
    logits = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    input_dict = {"logits": logits, "name": "log_softmax_mixed"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, small values
    logits = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    input_dict = {"logits": logits, "name": "log_softmax_small"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, with one extremely large number
    logits = np.array([[1.0, 2.0, 1000.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict = {"logits": logits, "name": "log_softmax_large_value"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, with one extremely small number
    logits = np.array([[1.0, 2.0, -1000.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"logits": logits, "name": "log_softmax_small_value"}
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
