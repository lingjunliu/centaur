
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_row_splits_to_segment_ids_inputs():
    list_of_inputs = []

    # Input 1
    splits = tf.constant([0, 3, 3, 5, 6, 9], dtype=tf.int64)
    name = None
    out_type = None
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    # Input 2
    splits = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    name = "segment_ids"
    out_type = tf.int32
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    # Input 3
    splits = tf.constant([0, 0, 0, 0], dtype=tf.int64)
    name = ""
    out_type = tf.int64
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    # Input 4
    splits = tf.constant([0, 10], dtype=tf.int32)
    name = "test_name"
    out_type = None
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    # Input 5
    splits = tf.constant([0, 1, 5, 10], dtype=tf.int64)
    name = None
    out_type = tf.int64
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    # Input 6
    splits = tf.constant([0, 2, 4, 6, 8, 10], dtype=tf.int32)
    name = "seg_ids"
    out_type = None
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    # Input 7
    splits = tf.constant([0, 1, 1, 1, 1], dtype=tf.int64)
    name = None
    out_type = tf.int32
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    # Input 8
    splits = tf.constant([0, 2, 2, 5], dtype=tf.int32)
    name = "segmentation"
    out_type = None
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    # Input 9
    splits = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.int64)
    name = None
    out_type = tf.int64
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    # Input 10
    splits = tf.constant([0, 5], dtype=tf.int32)
    name = "row_ids"
    out_type = tf.int32
    input_dict = {"splits": splits, "name": name, "out_type": out_type}
    list_of_inputs.append({"splits": splits.numpy(), "name": name, "out_type": out_type})

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.row_splits_to_segment_ids"] = tf_ragged_row_splits_to_segment_ids_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.row_splits_to_segment_ids' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.row_splits_to_segment_ids'.")

check_valid('tf.ragged.row_splits_to_segment_ids', generated_inputs['tf.ragged.row_splits_to_segment_ids'], lib="tf", suffix=0)
