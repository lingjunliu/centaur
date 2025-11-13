
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_nn_ctc_loss_inputs():
    list_of_inputs = []
    
    labels = np.array([[1, 2, 3, 0, 0], [1, 3, 0, 0, 0]], dtype=np.int64)
    logits = np.random.randn(10, 2, 5).astype(np.float32)
    label_length = np.array([3, 2], dtype=np.int64)
    logit_length = np.array([10, 10], dtype=np.int64)
    
    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": True,
        "unique": None,
        "blank_index": 0,
        "name": "ctc_loss_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    labels = np.array([[2, 3, 1, 0], [1, 2, 0, 0]], dtype=np.int64)
    logits = np.random.randn(2, 8, 6).astype(np.float32)
    label_length = np.array([3, 2], dtype=np.int64)
    logit_length = np.array([8, 8], dtype=np.int64)
    
    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": False,
        "unique": None,
        "blank_index": 0,
        "name": "ctc_loss_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    labels = np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int64)
    logits = np.random.randn(12, 2, 7).astype(np.float32)
    label_length = np.array([2, 1], dtype=np.int64)
    logit_length = np.array([12, 12], dtype=np.int64)
    
    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": True,
        "unique": None,
        "blank_index": 6,
        "name": "ctc_loss_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    labels = np.array([[1, 2, 3, 4, 0], [2, 3, 0, 0, 0]], dtype=np.int64)
    logits = np.random.randn(15, 2, 8).astype(np.float32)
    label_length = np.array([4, 2], dtype=np.int64)
    logit_length = np.array([15, 15], dtype=np.int64)
    
    input_dict = {
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": True,
        "unique": None,
        "blank_index": -1,
        "name": "ctc_loss_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    labels = np.array([[1, 0, 0, 0], [2, 3, 4, 0]], dtype=np.int64)
    logits = np.random.randn(3, 20, 10).astype(np.float32)
    label_length

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.ctc_loss_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.ctc_loss_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.ctc_loss', generated_inputs['tf.nn.ctc_loss_1'], lib="tf", suffix=1)
