
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_data_experimental_to_variant_inputs():
    list_of_inputs = []

    # Input 1: Simple dataset of integers
    dataset = tf.constant([1, 2, 3, 4, 5])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset of strings
    dataset = tf.constant(["a", "b", "c"])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset of tuples (homogenous type, tensors)
    dataset = tf.constant([(1, 2), (3, 4)])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset of numbers
    dataset = tf.constant([1.0, 2.0, 3.0])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(input_dict)
    
    # Input 5: Dataset using range
    dataset = tf.range(10)
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset of batched data
    dataset = tf.constant([1, 2, 3, 4, 5])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset).batch(2)}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with map (simplified lambda to avoid pickling issues)
    dataset = tf.constant([1, 2, 3])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset).map(lambda x: tf.cast(x * 2, tf.int64))}
    list_of_inputs.append(input_dict)

    # Input 8: Dataset with filter (simplified lambda to avoid pickling issues)
    dataset = tf.constant([1, 2, 3, 4, 5])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset).filter(lambda x: tf.math.equal(x % 2, 0))}
    list_of_inputs.append(input_dict)
    
    # Input 9: Prefetched Dataset
    dataset = tf.constant([1, 2, 3, 4, 5])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset).prefetch(buffer_size=tf.data.AUTOTUNE)}
    list_of_inputs.append(input_dict)

    # Input 10: Empty dataset
    dataset = tf.constant([])
    input_dict = {"dataset": tf.data.Dataset.from_tensor_slices(dataset)}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.to_variant"] = tf_data_experimental_to_variant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.to_variant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.to_variant'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.to_variant', generated_inputs['tf.data.experimental.to_variant'], lib="tf", suffix=0)
