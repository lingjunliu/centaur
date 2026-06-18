
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSoftmaxCrossEntropyWithLogits_inputs():
    list_of_inputs = []

    # Input 1
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 2], dtype=np.int32)
    name = "loss_1"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 2
    features = np.array([[-1.0, 0.0], [2.5, -3.2], [0.1, 0.2], [10.0, 11.0]], dtype=np.float64)
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    name = "loss_2"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 3
    features = np.arange(10, dtype=np.float16).reshape((1, 10))
    labels = np.array([5], dtype=np.int32)
    name = "loss_3"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 4
    features = np.random.uniform(-5.0, 5.0, (10, 5)).astype(np.float32)
    labels = np.random.randint(0, 5, size=(10,), dtype=np.int64)
    name = "loss_4"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 5
    features = np.zeros((3, 3), dtype=np.float64)
    labels = np.array([0, 1, 2], dtype=np.int32)
    name = "loss_5"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 6
    features = np.array([[100.0, -100.0, 0.0, 1.0], 
                         [-50.0, 50.0, 2.0, -2.0], 
                         [0.5, -0.5, 0.1, -0.1], 
                         [10.0, 20.0, 30.0, 40.0], 
                         [-1.0, -2.0, -3.0, -4.0]], dtype=np.float32)
    labels = np.array([2, 1, 3, 0, 2], dtype=np.int32)
    name = "loss_6"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 7
    features = np.eye(8, dtype=np.float16)
    labels = np.arange(8, dtype=np.int64)
    name = "loss_7"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 8
    features = np.random.normal(0.0, 1.0, (15, 2)).astype(np.float32)
    labels = np.random.randint(0, 2, size=(15,), dtype=np.int32)
    name = "loss_8"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 9
    features = np.ones((2, 20), dtype=np.float64)
    labels = np.array([19, 0], dtype=np.int64)
    name = "loss_9"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 10
    features = np.array([[1.5], [-2.3], [0.0], [9.1], [-0.5], [4.4]], dtype=np.float32)
    labels = np.zeros((6,), dtype=np.int32)
    name = "loss_10"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = tf_raw_ops_SparseSoftmaxCrossEntropyWithLogits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits', generated_inputs['tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits'], lib="tf", suffix=0)
