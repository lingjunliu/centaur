
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_in_top_k_inputs():
    list_of_inputs = []

    targets = np.array([0, 1, 3], dtype=np.int32)
    predictions = np.array([[1.2, -0.3, 2.8, 5.2], [0.1, 0.0, 0.0, 0.0], [0.0, 0.5, 0.3, 0.3]], dtype=np.float32)
    k = 2
    name = "test1"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([2, 0, 1], dtype=np.int64)
    predictions = np.array([[0.7, 0.9, 0.2], [0.1, 0.5, 0.8], [0.4, 0.6, 0.3]], dtype=np.float32)
    k = 1
    name = "test2"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([1, 2, 0], dtype=np.int32)
    predictions = np.array([[-0.1, 0.2, 0.3], [0.4, -0.5, 0.6], [0.7, 0.8, -0.9]], dtype=np.float32)
    k = 3
    name = "test3"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0], dtype=np.int64)
    predictions = np.array([[1.0]], dtype=np.float32)
    k = 1
    name = "test4"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([1, 1, 1], dtype=np.int32)
    predictions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    k = 2
    name = "test5"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0, 1], dtype=np.int64)
    predictions = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    k = 2
    name = "test6"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0, 0, 0], dtype=np.int32)
    predictions = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    k = 1
    name = "test7"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([2, 1, 0], dtype=np.int64)
    predictions = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)
    k = 4
    name = "test8"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0], dtype=np.int32)
    predictions = np.array([[-1.0, 2.0, 3.0]], dtype=np.float32)
    k = 2
    name = "test9"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([1, 2, 3], dtype=np.int64)
    predictions = np.array([[0.1, 0.9, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.2, 0.1, 0.9]], dtype=np.float32)
    k = 3
    name = "test10"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.in_top_k"] = tf_math_in_top_k_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.in_top_k' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.in_top_k'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.in_top_k', generated_inputs['tf.math.in_top_k'], lib="tf", suffix=0)
