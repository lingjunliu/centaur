
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_unique_inputs():
    list_of_inputs = []

    # Since tf.data.experimental.unique returns a dataset transformation function, 
    # it doesn't take any direct numpy inputs. We return an empty list of dictionaries.
    # The dictionaries are required by the framework, but their contents are irrelevant for this API.

    for _ in range(10):
        list_of_inputs.append({})

    return list_of_inputs

# generated_inputs = {} # This line was removed, according to the prompt it already exists
generated_inputs["tf.data.experimental.unique"] = tf_data_experimental_unique_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.unique'.")

check_valid('tf.data.experimental.unique', generated_inputs['tf.data.experimental.unique'], lib="tf", suffix=0)
