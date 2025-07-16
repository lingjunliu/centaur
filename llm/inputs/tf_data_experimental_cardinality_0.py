
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_data_experimental_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Dataset with a known cardinality
    dataset = tf.data.Dataset.range(10)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset with a different known cardinality
    dataset = tf.data.Dataset.range(1000)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset with infinite cardinality (repeat)
    dataset = tf.data.Dataset.range(5).repeat().take(10)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset with unknown cardinality (filter)
    dataset = tf.data.Dataset.range(10).filter(lambda x: x > 5).take(5) #Added take to avoid infinite shape analysis if it gets stuck
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset with unknown cardinality (map)
    dataset = tf.data.Dataset.range(10).map(lambda x: x * 2).take(5) #Added take
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset with unknown cardinality (flat_map)
    dataset = tf.data.Dataset.range(5).flat_map(lambda x: tf.data.Dataset.range(x)).take(5) #Added take
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with unknown cardinality (shuffle + filter)
    dataset = tf.data.Dataset.range(10).shuffle(buffer_size=10).filter(lambda x: x > 3).take(5) #Added take
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 8: Dataset with a known cardinality after applying a take operation
    dataset = tf.data.Dataset.range(20).take(5)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 9: Dataset with a known cardinality after applying a skip operation
    dataset = tf.data.Dataset.range(20).skip(5).take(5) # Add take
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 10: Dataset created from tensor slices.
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.cardinality"] = tf_data_experimental_cardinality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.cardinality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.cardinality'.")

check_valid('tf.data.experimental.cardinality', generated_inputs['tf.data.experimental.cardinality'], lib="tf", suffix=0)
