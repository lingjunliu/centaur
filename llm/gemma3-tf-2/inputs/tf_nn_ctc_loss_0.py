
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_ctc_loss_inputs():
    list_of_inputs = []

    input_dict_1 = {
        'labels': np.array([[1, 2, 3, 0]], dtype=np.int64),
        'logits': np.random.uniform(size=(5, 1, 4)).astype(np.float32),
        'label_length': np.array([3], dtype=np.int64),
        'logit_length': np.array([5], dtype=np.int64),
        'logits_time_major': True,
        'unique': np.array([0, 1, 2, 3], dtype=np.int64),
        'blank_index': 0,
        'name': "ctc_loss_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

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
