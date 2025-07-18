
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_lmdbreader_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.LMDBReader function.

    NOTE: The execution of this op fails with a `tensorflow.python.framework.errors_impl.UnimplementedError` 
    because LMDB support has been removed from recent versions of TensorFlow.
    There are no inputs that will allow this function to execute successfully in the
    target environment. Therefore, an empty list is returned to indicate that the
    API is effectively deprecated and unusable.
    """
    list_of_inputs = []
    
    return list_of_inputs

generated_inputs["tf.raw_ops.LMDBReader"] = tf_raw_ops_lmdbreader_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LMDBReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LMDBReader'.")

check_valid('tf.raw_ops.LMDBReader', generated_inputs['tf.raw_ops.LMDBReader'], lib="tf", suffix=0)
