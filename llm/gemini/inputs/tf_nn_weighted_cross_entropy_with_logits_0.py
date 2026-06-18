
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_weighted_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, pos_weight scalar
    labels = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    logits = np.array([1.5, -0.1, -10.0], dtype=np.float32)
    pos_weight = np.array(1.5, dtype=np.float32)
    name = "loss_1"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 2: 1D array, float32, pos_weight scalar < 1.0
    labels = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    logits = np.array([0.0, 2.0, -1.0], dtype=np.float32)
    pos_weight = np.array(0.5, dtype=np.float32)
    name = "loss_2"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 3: 2D array, float32, pos_weight scalar
    labels = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    logits = np.array([[-1.0, 1.0], [0.0, -2.0]], dtype=np.float32)
    pos_weight = np.array(2.0, dtype=np.float32)
    name = "loss_3"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 4: 2D array, float64, pos_weight scalar
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    logits = np.array([[10.0, -10.0], [-5.0, 5.0]], dtype=np.float64)
    pos_weight = np.array(1.0, dtype=np.float64)
    name = "loss_4"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 5: 3D array, float32, pos_weight 1D array (broadcastable)
    labels = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    logits = np.array([[[1.0, -1.0], [2.0, -2.0]]], dtype=np.float32)
    pos_weight = np.array([1.5, 0.5], dtype=np.float32)
    name = "loss_5"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 6: 1D array, float64, pos_weight broadcastable 1D array
    labels = np.array([0.0, 1.0, 0.5, 0.8], dtype=np.float64)
    logits = np.array([-100.0, 100.0, 0.0, -1.0], dtype=np.float64)
    pos_weight = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    name = "loss_6"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 7: 2D array, float32, pos_weight broadcastable 2D array
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    logits = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float32)
    pos_weight = np.array([[1.0], [2.0]], dtype=np.float32)
    name = "loss_7"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 8: 1D array with single element, float32
    labels = np.array([0.0], dtype=np.float32)
    logits = np.array([-0.5], dtype=np.float32)
    pos_weight = np.array([3.5], dtype=np.float32)
    name = "loss_8"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 9: 3D array, float64, pos_weight scalar
    labels = np.array([[[0.0], [1.0]], [[1.0], [0.0]]], dtype=np.float64)
    logits = np.array([[[0.1], [-0.1]], [[0.2], [-0.2]]], dtype=np.float64)
    pos_weight = np.array(0.1, dtype=np.float64)
    name = "loss_9"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 10: 2D array, float32, pos_weight matching shape
    labels = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    logits = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    pos_weight = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)
    name = "loss_10"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.nn.weighted_cross_entropy_with_logits"] = tf_weighted_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.weighted_cross_entropy_with_logits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.weighted_cross_entropy_with_logits'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.weighted_cross_entropy_with_logits', generated_inputs['tf.nn.weighted_cross_entropy_with_logits'], lib="tf", suffix=0)
