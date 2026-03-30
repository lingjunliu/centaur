
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomShuffleQueue_inputs():
    list_of_inputs = []

    # Input 1
    component_types = [tf.float32]
    shapes = []
    capacity = 10
    min_after_dequeue = 2
    seed = 0
    seed2 = 0
    container = ""
    shared_name = ""
    name = None
    input_dict = {"component_types": component_types, "shapes": shapes, "capacity": capacity, "min_after_dequeue": min_after_dequeue, "seed": seed, "seed2": seed2, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomShuffleQueue"] = tf_raw_ops_RandomShuffleQueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RandomShuffleQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomShuffleQueue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RandomShuffleQueue', generated_inputs['tf.raw_ops.RandomShuffleQueue'], lib="tf", suffix=0)
