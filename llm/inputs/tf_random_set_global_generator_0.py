
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_random_set_global_generator_inputs():
    list_of_inputs = []

    # Input 1
    generator = tf.random.Generator.from_seed(1234)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    generator = tf.random.Generator.from_non_deterministic_state()
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    generator = tf.random.Generator.from_seed(0)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.set_global_generator"] = tf_random_set_global_generator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.set_global_generator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.set_global_generator'.")

check_valid('tf.random.set_global_generator', generated_inputs['tf.random.set_global_generator'], lib="tf", suffix=0)
