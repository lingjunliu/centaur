
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_sparse_softmax_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    # Input 1: 1D labels, 2D logits, float32, int32
    labels = np.array([0, 2, 1], dtype=np.int32)
    logits = np.array([[2.0, -1.0, 0.5],
                       [0.0, 1.0, 3.0],
                       [-1.0, 2.0, -2.0]], dtype=np.float32)
    name = "loss_1"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 2: 2D labels, 3D logits, float64, int64
    labels = np.array([[0, 1], [2, 0]], dtype=np.int64)
    logits = np.array([[[1.5, -0.5, 2.0], [0.0, 2.5, -1.0]],
                       [[-2.0, 0.5, 1.0], [3.0, 1.0, 0.0]]], dtype=np.float64)
    name = "loss_2"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 3: 1D labels, 2D logits, float16, int32
    labels = np.array([1], dtype=np.int32)
    logits = np.array([[10.0, -10.0]], dtype=np.float16)
    name = "loss_3"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 4: Larger shapes, float32, negative values
    labels = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    logits = np.random.uniform(-5.0, 5.0, (5, 5)).astype(np.float32)
    name = "loss_4"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 5: Higher dimensions (3D labels, 4D logits), float32, int32
    labels = np.random.randint(0, 5, size=(2, 3, 4)).astype(np.int32)
    logits = np.random.normal(0.0, 1.0, size=(2, 3, 4, 5)).astype(np.float32)
    name = "loss_5"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 6: Minimal shapes, float32, int64
    labels = np.array([[1]], dtype=np.int64)
    logits = np.array([[[0.1, 0.9]]], dtype=np.float32)
    name = "loss_6"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 7: Larger logits classes, float64, int32
    labels = np.array([0, 9], dtype=np.int32)
    logits = np.zeros((2, 10), dtype=np.float64)
    name = "loss_7"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 8: Multi-dim (2D labels, 3D logits), float16, int64
    labels = np.array([[0, 2], [1, 1], [2, 0]], dtype=np.int64)
    logits = np.ones((3, 2, 3), dtype=np.float16)
    name = "loss_8"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 9: Large number of classes (100 classes), 1D labels, float32
    labels = np.array([42, 99], dtype=np.int32)
    logits = np.random.standard_normal((2, 100)).astype(np.float32)
    name = "loss_9"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 10: Higher dimension 3D, float64, int64
    labels = np.random.randint(0, 2, size=(2, 2, 2)).astype(np.int64)
    logits = np.random.uniform(-10.0, 10.0, size=(2, 2, 2, 2)).astype(np.float64)
    name = "loss_10"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    return list_of_inputs

generated_inputs["tf.nn.sparse_softmax_cross_entropy_with_logits"] = tf_nn_sparse_softmax_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.sparse_softmax_cross_entropy_with_logits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.sparse_softmax_cross_entropy_with_logits'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.sparse_softmax_cross_entropy_with_logits', generated_inputs['tf.nn.sparse_softmax_cross_entropy_with_logits'], lib="tf", suffix=0)
