
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_priority_queue_inputs():
    list_of_inputs = []

    # Input 1: Empty shapes and default component_types
    input_dict = {
        "shapes": [],
        "component_types": [],
        "capacity": -1,
        "container": "",
        "shared_name": "",
        "name": "queue_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.PriorityQueue"] = tf_raw_ops_priority_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.PriorityQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PriorityQueue'.")

check_valid('tf.raw_ops.PriorityQueue', generated_inputs['tf.raw_ops.PriorityQueue'], lib="tf", suffix=0)
