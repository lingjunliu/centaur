
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_train_coordinator_inputs():
    list_of_inputs = []

    # Input 1: Empty tuple
    input_dict = {'clean_stop_exception_types': ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple with one exception type (BaseException)
    input_dict = {'clean_stop_exception_types': (BaseException,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple with one exception type (ValueError)
    input_dict = {'clean_stop_exception_types': (ValueError,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple with two exception types (ValueError, TypeError)
    input_dict = {'clean_stop_exception_types': (ValueError, TypeError)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple with three exception types (ValueError, TypeError, KeyError)
    input_dict = {'clean_stop_exception_types': (ValueError, TypeError, KeyError)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.train.Coordinator"] = tf_train_coordinator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for input_dict in list_of_inputs:
        try:
            _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        except Exception as e:
            print(f"Error during validity check for {api}: {e}")
            raise

def get_signature(api, lib="tf", suffix=0):
    if api == 'tf.train.Coordinator':
        return {'clean_stop_exception_types': 'tuple'}
    raise ValueError(f"Unknown API: {api}")

def get_abstract_input(concrete, signature):
    abstract = {}
    for arg, domain in signature.items():
        abstract[arg] = get_ll(domain, concrete[arg])
    return abstract

def get_ll(domain, value):
    if domain == 'tuple':
        return 'tuple'
    raise ValueError(f"Unknown domain: {domain}")

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.train.Coordinator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.train.Coordinator'.")

check_valid('tf.train.Coordinator', generated_inputs['tf.train.Coordinator'], lib="tf", suffix=0)
