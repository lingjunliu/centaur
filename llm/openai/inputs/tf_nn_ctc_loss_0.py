
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(123)

def tf_nn_ctc_loss_inputs():
    def actual_blank_index(num_labels, blank_index):
        return num_labels + blank_index if blank_index is not None and blank_index < 0 else blank_index

    def generate_labels(batch_size, max_label_len, num_labels, label_len_arr, blank_index, out_dtype=np.int64):
        bidx = actual_blank_index(num_labels, blank_index)
        allowed_vals = [i for i in range(num_labels) if i != bidx]
        if len(allowed_vals) == 0:
            allowed_vals = [0]
        labels = np.zeros((batch_size, max_label_len), dtype=out_dtype)
        for b in range(batch_size):
            L = int(label_len_arr[b])
            if L > 0:
                labels[b, :L] = np.random.choice(allowed_vals, size=L)
        return labels

    list_of_inputs = []

    # Input 1
    B, Lmax, T, C = 4, 5, 12, 6
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int64)
    logit_length = np.full((B,), T, dtype=np.int64)
    logits_time_major = True
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = np.random.randn(T, B, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case1"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 2
    B, Lmax, T, C = 2, 4, 7, 5
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int32)
    logit_length = np.array([7, 5], dtype=np.int32)
    logits_time_major = False
    blank_index = -1
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = np.random.randn(B, T, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case2"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 3
    B, Lmax, T, C = 4, 6, 10, 7
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int64)
    logit_length = np.array([10, 9, 8, 10], dtype=np.int64)
    logits_time_major = True
    blank_index = C - 1
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = np.random.randn(T, B, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case3"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 4
    B, Lmax, T, C = 3, 4, 6, 5
    label_length = np.array([0, 2, 3], dtype=np.int32)
    logit_length = np.array([6, 6, 5], dtype=np.int32)
    logits_time_major = True
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = np.random.uniform(-1.0, 1.0, size=(T, B, C)).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case4_zero_label_len"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 5
    B, Lmax, T, C = 1, 3, 4, 4
    label_length = np.array([1], dtype=np.int64)
    logit_length = np.array([4], dtype=np.int64)
    logits_time_major = False
    blank_index = -1
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = (np.random.randn(B, T, C) * 0.5).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case5_single_sample"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 6
    B, Lmax, T, C = 5, 8, 12, 10
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int32)
    logit_length = np.maximum(label_length + 1, np.random.randint(Lmax, T + 1, size=(B,), dtype=np.int32))
    logits_time_major = True
    blank_index = -2
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = np.random.randn(T, B, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case6_negative_blank_two"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 7
    B, Lmax, T, C = 2, 2, 4, 3
    label_length = np.array([2, 1], dtype=np.int64)
    logit_length = np.array([4, 3], dtype=np.int64)
    logits_time_major = False
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = np.random.uniform(-2.0, 2.0, size=(B, T, C)).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case7_small_dims"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 8
    B, Lmax, T, C = 6, 7, 15, 8
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int32)
    logit_length = np.random.randint(Lmax, T + 1, size=(B,), dtype=np.int32)
    logits_time_major = True
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = (np.random.randn(T, B, C) * 1.5).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case8_large_batch"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 9
    B, Lmax, T, C = 2, 3, 5, 4
    label_length = np.array([3, 2], dtype=np.int32)
    logit_length = np.array([5, 5], dtype=np.int32)
    logits_time_major = False
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = np.random.randn(B, T, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case9_float32_logits"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 10
    B, Lmax, T, C = 3, 5, 9, 5
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int64)
    logit_length = np.array([9, 8, 7], dtype=np.int64)
    logits_time_major = True
    blank_index = 2
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = np.random.randn(T, B, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case10_mid_blank"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

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
