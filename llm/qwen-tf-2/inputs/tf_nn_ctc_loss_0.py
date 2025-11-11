
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def generate_ctc_loss_inputs():
    list_of_inputs = []
    
    # Input 1: Basic dense labels with logits time major=True
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    logits = np.random.rand(3, 2, 7).astype(np.float32)
    label_length = np.array([3, 3], dtype=np.int32)
    logit_length = np.array([3, 3], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 0
    name = "ctc_loss_dense"
    
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
    
    # Input 2: Sparse labels with logits time major=False
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    logits = np.random.rand(2, 3, 7).astype(np.float32)
    label_length = np.array([3, 3], dtype=np.int32)
    logit_length = np.array([3, 3], dtype=np.int32)
    logits_time_major = False
    unique = None
    blank_index = 0
    name = "ctc_loss_sparse"
    
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
    
    # Input 3: Dense labels with different blank index
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    logits = np.random.rand(3, 2, 7).astype(np.float32)
    label_length = np.array([3, 3], dtype=np.int32)
    logit_length = np.array([3, 3], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 1
    name = "ctc_loss_blank_1"
    
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
    
    # Input 4: Dense labels with negative blank index
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    logits = np.random.rand(3, 2, 7).astype(np.float32)
    label_length = np.array([3, 3], dtype=np.int32)
    logit_length = np.array([3, 3], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = -1
    name = "ctc_loss_blank_neg"
    
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
    
    # Input 5: Dense labels with unique label indices
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    logits = np.random.rand(3, 2, 7).astype(np.float32)
    label_length = np.array([3, 3], dtype=np.int32)
    logit_length = np.array([3, 3], dtype=np.int32)
    logits_time_major = True
    unique = np.array([1, 2, 3], dtype=np.int32)
    blank_index = 0
    name = "ctc_loss_unique"
    
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
    
    # Input 6: Different batch sizes and label lengths
    labels = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    logits = np.random.rand(4, 2, 9).astype(np.float32)
    label_length = np.array([4, 4], dtype=np.int32)
    logit_length = np.array([4, 4], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 0
    name = "ctc_loss_varied_batch"
    
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
    
    # Input 7: Different number of frames and label lengths
    labels = np.array([[1, 2], [3, 4]], dtype=np.int32)
    logits = np.random.rand(5, 2, 6).astype(np.float32)
    label_length = np.array([2, 2], dtype=np.int32)
    logit_length = np.array([5, 5], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 0
    name = "ctc_loss_varied_frames"
    
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
    
    # Input 8: Different number of labels and label lengths
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    logits = np.random.rand(3, 2, 8).astype(np.float32)
    label_length = np.array([3, 3], dtype=np.int32)
    logit_length = np.array([3, 3], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 0
    name = "ctc_loss_varied_labels"
    
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
    
    # Input 9: Negative values in labels
    labels = np.array([[1, -2, 3], [-4, 5, 6]], dtype=np.int32)
    logits = np.random.rand(3, 2, 7).astype(np.float32)
    label_length = np.array([3, 3], dtype=np.int32)
    logit_length = np.array([3, 3], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 0
    name = "ctc_loss_negative_labels"
    
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
    
    # Input 10: Different label lengths and logit lengths
    labels = np.array([[1, 2], [3, 4]], dtype=np.int32)
    logits = np.random.rand(3, 2, 6).astype(np.float32)
    label_length = np.array([2, 2], dtype=np.int32)
    logit_length = np.array([3, 3], dtype=np.int32)
    logits_time_major = True
    unique = None
    blank_index = 0
    name = "ctc_loss_mismatched_lengths"
    
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

generated_inputs["tf.nn.ctc_loss"] = generate_ctc_loss_inputs()

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
