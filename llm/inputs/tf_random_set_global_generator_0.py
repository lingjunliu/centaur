
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_random_set_global_generator_inputs():
    list_of_inputs = []

    # Input 1: Basic Generator
    generator = tf.random.Generator.from_seed(123)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Generator with a different seed
    generator = tf.random.Generator.from_seed(456)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Generator with a large seed
    generator = tf.random.Generator.from_seed(2**32 - 1)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Newly created generator.
    generator = tf.random.Generator.from_non_deterministic_state()
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Generator created from split. Ensure at least two generators are created.
    gen1 = tf.random.Generator.from_seed(10)
    splits = gen1.split() # Split into 2 generators, returns a tuple or list (implementation detail)
    generator = splits[0]
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another generator from split
    gen1 = tf.random.Generator.from_seed(20)
    splits = gen1.split()
    generator = splits[0] # Using index 0 again as a safe choice.
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different seed.
    generator = tf.random.Generator.from_seed(999)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Generator with specific algorithm
    generator = tf.random.Generator.from_seed(123, alg='philox')
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Generator with specific algorithm
    generator = tf.random.Generator.from_seed(456, alg='threefry')
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A generator that has generated some random numbers
    generator = tf.random.Generator.from_seed(1000)
    generator.normal(shape=(5, 5))
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
