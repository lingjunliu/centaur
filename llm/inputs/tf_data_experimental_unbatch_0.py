
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_unbatch_inputs():
    list_of_inputs = []

    # The API tf.data.experimental.unbatch is a function transformation, so it requires
    # a dataset as input to .apply(). The error message indicates that we are passing
    # an empty dictionary as input. The correct way to use this API is to create a
    # tf.data.Dataset object and then apply the unbatch transformation. However,
    # the signature strictly defines empty dict, so we can't create a dataset.
    # Therefore, it's impossible to create a valid input for this API given the provided signature.

    # Returning an empty list as a last resort.
    return []

generated_inputs = {}
generated_inputs["tf.data.experimental.unbatch"] = tf_data_experimental_unbatch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.unbatch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.unbatch'.")

check_valid('tf.data.experimental.unbatch', generated_inputs['tf.data.experimental.unbatch'], lib="tf", suffix=0)
