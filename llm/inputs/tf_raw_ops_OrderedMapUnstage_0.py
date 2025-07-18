
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_orderedmapunstage_inputs():
    """
    NOTE: The tf.raw_ops.OrderedMapUnstage operation is a blocking op.
    It is designed to wait until data is staged by a corresponding
    tf.raw_ops.OrderedMapStage op. When executed in isolation without
    a preceding stage op, it will intentionally block and cause a timeout.
    This is the expected runtime behavior. The input provided here is
    syntactically valid for the operation itself.
    """
    list_of_inputs = []

    # Input 1: A single, minimal, syntactically valid input. The timeout is
    # an expected runtime behavior due to the blocking nature of the op.
    input_dict = {
        'key': np.array(1, dtype=np.int64),
        'indices': np.array([0], dtype=np.int32),
        'dtypes': [np.float32],
        'capacity': 0,
        'memory_limit': 0,
        'container': '',
        'shared_name': '',
        'name': 'MinimalUnstage'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.OrderedMapUnstage"] = tf_raw_ops_orderedmapunstage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.OrderedMapUnstage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapUnstage'.")

check_valid('tf.raw_ops.OrderedMapUnstage', generated_inputs['tf.raw_ops.OrderedMapUnstage'], lib="tf", suffix=0)
