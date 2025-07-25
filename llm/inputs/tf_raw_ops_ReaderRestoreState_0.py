
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_reader_restore_state_inputs():
    list_of_inputs = []

    # The API `tf.raw_ops.ReaderRestoreState` is fundamentally incompatible
    # with TensorFlow's eager execution mode. The operation requires a `ref`
    # tensor for its `reader_handle`, a concept from TF1's graph mode which
    # is not supported in the eager context. Any call to this function in
    # eager mode will intentionally raise a `RuntimeError`.
    #
    # The following single input is syntactically correct according to the
    # API signature. It is provided as the simplest possible valid case,
    # acknowledging that the runtime error is unavoidable.

    # Input 1: A minimal, syntactically valid input
    input_dict = {
        'reader_handle': np.array('a_reader_handle', dtype=object),
        'state': np.array('a_serialized_state', dtype=object),
        'name': 'MinimalRestoreOp'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ReaderRestoreState"] = tf_raw_ops_reader_restore_state_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReaderRestoreState' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderRestoreState'.")

check_valid('tf.raw_ops.ReaderRestoreState', generated_inputs['tf.raw_ops.ReaderRestoreState'], lib="tf", suffix=0)
