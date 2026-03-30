
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_get_single_element_inputs():
    list_of_inputs = []

    # Input 1
    dataset = tf.data.Dataset.from_tensors(np.array([1, 2, 3])).batch(3).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 2
    dataset = tf.data.Dataset.from_tensors(np.array([[1, 2], [3, 4]])).batch(2).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 3
    dataset = tf.data.Dataset.from_tensors(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])).batch(2).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 4
    dataset = tf.data.Dataset.from_tensors(np.array([1.0, 2.0, 3.0])).batch(3).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 5
    dataset = tf.data.Dataset.from_tensors(np.array([[-1, -2], [-3, -4]])).batch(2).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 6
    dataset = tf.data.Dataset.from_tensors(np.array([True, False, True])).batch(3).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 7
    dataset = tf.data.Dataset.from_tensors(np.array(["hello", "world"])).batch(2).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 8
    dataset = tf.data.Dataset.from_tensors(np.array([1])).batch(1).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 9
    dataset = tf.data.Dataset.from_tensors(np.array([np.float32(1.0), np.float32(2.0)])).batch(2).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 10
    dataset = tf.data.Dataset.from_tensors(np.array([np.int64(1), np.int64(2)])).batch(2).take(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.get_single_element"] = tf_data_experimental_get_single_element_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.get_single_element' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.get_single_element'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.get_single_element', generated_inputs['tf.data.experimental.get_single_element'], lib="tf", suffix=0)
