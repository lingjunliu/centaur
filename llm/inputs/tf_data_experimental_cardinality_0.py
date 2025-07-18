
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_cardinality_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.cardinality.
    The input must be a tf.data.Dataset object for the API call to succeed.
    """
    list_of_inputs = []

    # Input 1: Finite, known cardinality from range
    dataset1 = tf.data.Dataset.range(42)
    input_dict1 = {'dataset': dataset1}
    list_of_inputs.append(input_dict1)

    # Input 2: Finite, known cardinality from a numpy array
    dataset2 = tf.data.Dataset.from_tensor_slices(np.arange(100, dtype=np.int64))
    input_dict2 = {'dataset': dataset2}
    list_of_inputs.append(input_dict2)

    # Input 3: Finite, known cardinality of 1 from from_tensors
    dataset3 = tf.data.Dataset.from_tensors(tf.constant([1, 2, 3, 4], dtype=tf.float32))
    input_dict3 = {'dataset': dataset3}
    list_of_inputs.append(input_dict3)

    # Input 4: Zero cardinality from an empty numpy array
    dataset4 = tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.int32))
    input_dict4 = {'dataset': dataset4}
    list_of_inputs.append(input_dict4)

    # Input 5: Infinite cardinality using repeat()
    dataset5 = tf.data.Dataset.range(10).repeat()
    input_dict5 = {'dataset': dataset5}
    list_of_inputs.append(input_dict5)

    # Input 6: Unknown cardinality after using filter()
    dataset6 = tf.data.Dataset.range(200).filter(lambda x: x % 2 == 0)
    input_dict6 = {'dataset': dataset6}
    list_of_inputs.append(input_dict6)

    # Input 7: Unknown cardinality from a Python generator
    def simple_generator():
        for i in range(25):
            yield i
    dataset7 = tf.data.Dataset.from_generator(simple_generator, output_signature=tf.TensorSpec(shape=(), dtype=tf.int32))
    input_dict7 = {'dataset': dataset7}
    list_of_inputs.append(input_dict7)

    # Input 8: Finite, known cardinality from zip()
    ds8a = tf.data.Dataset.range(15)
    ds8b = tf.data.Dataset.from_tensor_slices(np.random.rand(15, 3))
    dataset8 = tf.data.Dataset.zip((ds8a, ds8b))
    input_dict8 = {'dataset': dataset8}
    list_of_inputs.append(input_dict8)

    # Input 9: Finite, known cardinality from concatenate()
    ds9a = tf.data.Dataset.from_tensor_slices(np.ones(5))
    ds9b = tf.data.Dataset.from_tensor_slices(np.zeros(10))
    dataset9 = ds9a.concatenate(ds9b)
    input_dict9 = {'dataset': dataset9}
    list_of_inputs.append(input_dict9)

    # Input 10: Infinite cardinality from concatenating with an infinite dataset
    ds10a = tf.data.Dataset.range(50)
    ds10b = tf.data.Dataset.range(1).repeat()
    dataset10 = ds10a.concatenate(ds10b)
    input_dict10 = {'dataset': dataset10}
    list_of_inputs.append(input_dict10)

    return list_of_inputs

generated_inputs["tf.data.experimental.cardinality"] = tf_data_experimental_cardinality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.cardinality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.cardinality'.")

check_valid('tf.data.experimental.cardinality', generated_inputs['tf.data.experimental.cardinality'], lib="tf", suffix=0)
