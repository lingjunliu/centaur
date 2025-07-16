
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_random_set_global_generator_inputs():
    list_of_inputs = []

    # Input 1: Basic Generator
    generator = tf.random.Generator.from_seed(123)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Generator with different seed
    generator = tf.random.Generator.from_seed(42)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Generator created from split
    initial_generator = tf.random.Generator.from_seed(7)
    splits = initial_generator.split()
    generator = splits[0]
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Another generator created from split. Check length.
    initial_generator = tf.random.Generator.from_seed(15)
    splits = initial_generator.split()
    if len(splits) > 1:
        generator = splits[1]
    else:
        generator = tf.random.Generator.from_seed(16)  # Use a default generator if split fails
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Generator after some random number generation
    generator = tf.random.Generator.from_seed(22)
    #generator.normal(shape=(2, 2)) #Removed operation, Generator doesn't have shape
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Generator after reset
    generator = tf.random.Generator.from_seed(30)
    generator.reset_from_seed(40)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Generator with large seed
    generator = tf.random.Generator.from_seed(2**31 - 1)
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Generator using different algorithm
    generator = tf.random.Generator.from_seed(10, alg='philox')
    input_dict = {"generator": generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Generator from a saved state
    generator = tf.random.Generator.from_seed(50)
    state = generator.state
    new_generator = tf.random.Generator.from_state(state, alg=generator.algorithm)
    input_dict = {"generator": new_generator}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Generator with same seed as global generator (before setting)
    global_generator_before = tf.random.get_global_generator()
    #seed_before = global_generator_before.seed  # Removed access to non-existent attribute
    generator = tf.random.Generator.from_seed(60) # Use a different seed instead

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
