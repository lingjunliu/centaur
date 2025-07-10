
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_ctc_loss_inputs():
    list_of_inputs = []

    # Input 1
    labels = np.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=np.int64)
    logits = np.random.rand(7, 2, 6).astype(np.float32)
    label_length = np.array([3, 2], dtype=np.int32)
    logit_length = np.array([7, 7], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 0
    name = "ctc_loss_1"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    labels = np.array([[1, 2, 3, 0], [4, 5, 1, 0]], dtype=np.int64)
    logits = np.random.rand(2, 5, 6).astype(np.float32)
    label_length = np.array([3, 3], dtype=np.int32)
    logit_length = np.array([5, 5], dtype=np.int32)
    logits_time_major = False
    unique = None
    blank_index = 0
    name = "ctc_loss_2"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    labels = np.array([[1, 2, 3], [4, 5, 1]], dtype=np.int64)
    logits = np.random.rand(5, 2, 6).astype(np.float32)
    label_length = np.array([3, 3], dtype=np.int32)
    logit_length = np.array([5, 5], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 1
    name = "ctc_loss_3"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    labels = np.array([[1, 2], [4, 5]], dtype=np.int64)
    logits = np.random.rand(2, 4, 6).astype(np.float32)
    label_length = np.array([2, 2], dtype=np.int32)
    logit_length = np.array([4, 4], dtype=np.int32)
    logits_time_major = False
    unique = None
    blank_index = -1
    name = "ctc_loss_4"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    labels = np.array([[1, 2, 0], [4, 0, 0]], dtype=np.int64)
    logits = np.random.rand(5, 2, 3).astype(np.float32)
    label_length = np.array([2, 1], dtype=np.int32)
    logit_length = np.array([5, 5], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 0
    name = "ctc_loss_5"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    labels = np.array([[1, 2, 3, 4], [5, 1, 2, 0]], dtype=np.int64)
    logits = np.random.rand(2, 6, 6).astype(np.float32)
    label_length = np.array([4, 3], dtype=np.int32)
    logit_length = np.array([6, 6], dtype=np.int32)
    logits_time_major = False
    unique = None
    blank_index = 2
    name = "ctc_loss_6"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    labels = np.array([[1, 2], [4, 5]], dtype=np.int64)
    logits = np.random.rand(4, 2, 3).astype(np.float32)
    label_length = np.array([2, 2], dtype=np.int32)
    logit_length = np.array([4, 4], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = -2
    name = "ctc_loss_7"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    labels = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 0]], dtype=np.int64)
    logits = np.random.rand(8, 2, 7).astype(np.float32)
    label_length = np.array([5, 4], dtype=np.int32)
    logit_length = np.array([8, 8], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 0
    name = "ctc_loss_8"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    labels = np.array([[1, 2], [4, 5]], dtype=np.int64)
    logits = np.random.rand(2, 3, 4).astype(np.float32)
    label_length = np.array([2, 2], dtype=np.int32)
    logit_length = np.array([3, 3], dtype=np.int32)
    logits_time_major = False
    unique = None
    blank_index = 0
    name = "ctc_loss_9"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    labels = np.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=np.int64)
    logits = np.random.rand(7, 2, 6).astype(np.float32)
    label_length = np.array([3, 2], dtype=np.int32)
    logit_length = np.array([7, 7], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 5
    name = "ctc_loss_10"

    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.ctc_loss"] = tf_nn_ctc_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.ctc_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.ctc_loss'.")

check_valid('tf.nn.ctc_loss', generated_inputs['tf.nn.ctc_loss'], lib="tf", suffix=0)
