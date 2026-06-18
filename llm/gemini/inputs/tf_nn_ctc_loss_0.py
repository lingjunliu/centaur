
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_nn_ctc_loss_inputs():
    list_of_inputs = []

    # Input 1
    labels = np.array([[1, 2, 0], [2, 1, 1]], dtype=np.int32)
    logits = np.random.uniform(size=(5, 2, 4)).astype(np.float32)
    label_length = np.array([2, 3], dtype=np.int32)
    logit_length = np.array([5, 5], dtype=np.int32)
    logits_time_major = True
    unique = np.array([], dtype=np.int32)
    blank_index = 0
    name = "ctc_loss_1"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 2
    labels = np.array([[1, 2, 3, 0, 0], [4, 5, 1, 2, 0], [3, 2, 1, 4, 5], [1, 0, 0, 0, 0]], dtype=np.int64)
    logits = np.random.uniform(size=(4, 10, 8)).astype(np.float32)
    label_length = np.array([3, 4, 5, 1], dtype=np.int64)
    logit_length = np.array([10, 10, 10, 10], dtype=np.int64)
    logits_time_major = False
    unique = np.array([], dtype=np.int64)
    blank_index = -1
    name = "ctc_loss_2"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 3
    labels = np.array([[1, 2]], dtype=np.int32)
    logits = np.random.uniform(size=(4, 1, 3)).astype(np.float32)
    label_length = np.array([2], dtype=np.int32)
    logit_length = np.array([4], dtype=np.int32)
    logits_time_major = True
    unique = np.array([], dtype=np.int32)
    blank_index = 0
    name = "ctc_loss_3"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 4
    labels = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 1, 0]], dtype=np.int32)
    logits = np.random.uniform(size=(3, 15, 6)).astype(np.float32)
    label_length = np.array([4, 4, 3], dtype=np.int32)
    logit_length = np.array([15, 15, 15], dtype=np.int32)
    logits_time_major = False
    unique = np.array([], dtype=np.int32)
    blank_index = 5
    name = "ctc_loss_4"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 5
    labels = np.array([[1, 2, 3, 4, 5, 0], [5, 4, 3, 2, 1, 0], [2, 3, 4, 0, 0, 0]], dtype=np.int64)
    logits = np.random.uniform(size=(12, 3, 8)).astype(np.float32)
    label_length = np.array([5, 5, 3], dtype=np.int64)
    logit_length = np.array([12, 12, 12], dtype=np.int64)
    logits_time_major = True
    unique = np.array([], dtype=np.int64)
    blank_index = 0
    name = "ctc_loss_5"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 6
    labels = np.random.randint(1, 5, size=(16, 8)).astype(np.int32)
    logits = np.random.uniform(size=(16, 20, 10)).astype(np.float32)
    label_length = np.random.randint(4, 9, size=16).astype(np.int32)
    logit_length = np.random.randint(15, 21, size=16).astype(np.int32)
    logits_time_major = False
    unique = np.array([], dtype=np.int32)
    blank_index = 0
    name = "ctc_loss_6"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 7
    labels = np.random.randint(1, 15, size=(5, 10)).astype(np.int32)
    logits = np.random.uniform(size=(30, 5, 20)).astype(np.float32)
    label_length = np.array([8, 9, 7, 10, 6], dtype=np.int32)
    logit_length = np.array([28, 29, 30, 27, 26], dtype=np.int32)
    logits_time_major = True
    unique = np.array([], dtype=np.int32)
    blank_index = 19
    name = "ctc_loss_7"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 8
    labels = np.array([[0], [0]], dtype=np.int64)
    logits = np.random.uniform(size=(2, 2, 2)).astype(np.float32)
    label_length = np.array([1, 1], dtype=np.int64)
    logit_length = np.array([2, 2], dtype=np.int64)
    logits_time_major = False
    unique = np.array([], dtype=np.int64)
    blank_index = 1
    name = "ctc_loss_8"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 9
    labels = np.random.randint(1, 5, size=(10, 5)).astype(np.int32)
    logits = np.random.uniform(size=(15, 10, 6)).astype(np.float32)
    label_length = np.array([4, 3, 5, 4, 3, 5, 4, 3, 5, 4], dtype=np.int32)
    logit_length = np.array([15] * 10, dtype=np.int32)
    logits_time_major = True
    unique = np.array([], dtype=np.int32)
    blank_index = -1
    name = "ctc_loss_9"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 10
    labels = np.random.randint(1, 3, size=(4, 3)).astype(np.int64)
    logits = np.random.uniform(size=(4, 6, 4)).astype(np.float32)
    label_length = np.array([3, 2, 3, 2], dtype=np.int64)
    logit_length = np.array([6, 5, 6, 5], dtype=np.int64)
    logits_time_major = False
    unique = np.array([], dtype=np.int64)
    blank_index = 0
    name = "ctc_loss_10"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.nn.ctc_loss"] = tf_nn_ctc_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.ctc_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.ctc_loss'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.ctc_loss', generated_inputs['tf.nn.ctc_loss'], lib="tf", suffix=0)
