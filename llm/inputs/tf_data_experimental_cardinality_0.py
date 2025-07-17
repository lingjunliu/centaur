
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Dataset with known cardinality
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5]))
    input_dict1 = {"dataset": dataset1}
    list_of_inputs.append(input_dict1)

    # Input 2: Dataset from a generator
    def generator():
      for i in range(7):
        yield np.array(i)
    dataset2 = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(), dtype=tf.int64))
    input_dict2 = {"dataset": dataset2}
    list_of_inputs.append(input_dict2)

    # Input 3: Dataset with a map transformation
    dataset3 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4])).map(lambda x: x * 2)
    input_dict3 = {"dataset": dataset3}
    list_of_inputs.append(input_dict3)
    
    # Input 4: Dataset created from a list
    dataset4 = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5,6]))
    input_dict4 = {"dataset": dataset4}
    list_of_inputs.append(input_dict4)
    
    # Input 5: Dataset with take transformation
    dataset5 = tf.data.Dataset.from_tensor_slices(np.arange(15)).take(5)
    input_dict5 = {"dataset": dataset5}
    list_of_inputs.append(input_dict5)

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
