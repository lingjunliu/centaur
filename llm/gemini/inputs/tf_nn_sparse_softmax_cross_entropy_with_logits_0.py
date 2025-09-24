
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_sparse_softmax_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    logits = tf.constant([[2.0, -5.0, 0.5, -0.1], [0.0, 0.0, 1.9, 1.4], [-100.0, 100.0, -100.0, -100.0]], dtype=tf.float32).numpy()
    labels = tf.constant([0, 3, 1], dtype=tf.int32).numpy()
    name = "basic_example"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different batch size
    logits = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=tf.float32).numpy()
    labels = tf.constant([0, 1, 2, 0], dtype=tf.int32).numpy()
    name = "different_batch_size"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different number of classes
    logits = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=tf.float32).numpy()
    labels = tf.constant([2, 4], dtype=tf.int32).numpy()
    name = "different_num_classes"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative logits
    logits = tf.constant([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=tf.float32).numpy()
    labels = tf.constant([0, 1], dtype=tf.int32).numpy()
    name = "negative_logits"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger logits values
    logits = tf.constant([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=tf.float32).numpy()
    labels = tf.constant([0, 1], dtype=tf.int32).numpy()
    name = "larger_logits_values"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16 logits
    logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
    labels = tf.constant([0, 1], dtype=tf.int32).numpy()
    name = "float16_logits"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 labels
    logits = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32).numpy()
    labels = tf.constant([0, 1], dtype=tf.int64).numpy()
    name = "int64_labels"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D logits and 2D labels
    logits = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float32).numpy()
    labels = tf.constant([[0, 1], [1, 0]], dtype=tf.int32).numpy()
    name = "3d_logits_2d_labels"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single example
    logits = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32).numpy()
    labels = tf.constant([1], dtype=tf.int32).numpy()
    name = "single_example"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different values for labels
    logits = tf.constant([[2.0, -5.0, 0.5, -0.1], [0.0, 0.0, 1.9, 1.4], [-100.0, 100.0, -100.0, -100.0]], dtype=tf.float32).numpy()
    labels = tf.constant([3, 0, 1], dtype=tf.int32).numpy()
    name = "different_labels"
    input_dict = {"labels": labels, "logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.sparse_softmax_cross_entropy_with_logits"] = tf_nn_sparse_softmax_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.sparse_softmax_cross_entropy_with_logits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.sparse_softmax_cross_entropy_with_logits'.")

check_valid('tf.nn.sparse_softmax_cross_entropy_with_logits', generated_inputs['tf.nn.sparse_softmax_cross_entropy_with_logits'], lib="tf", suffix=0)
