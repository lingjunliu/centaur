
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_graph_inputs():
    list_of_inputs = []

    # Input 1: Empty graph
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another empty graph
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Yet another empty graph
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Empty graph for demonstration
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty graph
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty graph
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Empty graph
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty graph
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty Graph
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty Graph
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.Graph"] = tf_graph_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.Graph' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.Graph'.")

check_valid('tf.Graph', generated_inputs['tf.Graph'], lib="tf", suffix=0)
