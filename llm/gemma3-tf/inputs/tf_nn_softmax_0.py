
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_softmax_inputs():
    list_of_inputs = []

    logits = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    axis = -1
    name = "softmax_1"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 1
    name = "softmax_2"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
    axis = 0
    name = "softmax_3"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]], dtype=np.float64)
    axis = -1
    name = "softmax_4"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = 0
    name = "softmax_5"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float32)
    axis = 0
    name = "softmax_6"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 2
    name = "softmax_7"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    axis = -1
    name = "softmax_8"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    axis = -1
    name = "softmax_9"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    axis = 1
    name = "softmax_10"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.softmax"] = tf_nn_softmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.softmax'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.softmax', generated_inputs['tf.nn.softmax'], lib="tf", suffix=0)
