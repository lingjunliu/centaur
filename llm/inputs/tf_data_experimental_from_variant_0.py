
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_from_variant_inputs():
    list_of_inputs = []

    # Input 1
    variant = tf.constant(np.array(1, dtype=np.int32))
    structure = [tf.TensorSpec(shape=(), dtype=tf.int32)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    variant = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    structure = [tf.TensorSpec(shape=(3,), dtype=tf.int32)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    variant = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    structure = [tf.TensorSpec(shape=(2, 2), dtype=tf.int32)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    variant = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    structure = [tf.TensorSpec(shape=(3,), dtype=tf.float32)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    variant = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    structure = [tf.TensorSpec(shape=(2, 2), dtype=tf.float32)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    variant = tf.constant(np.array([True, False, True], dtype=np.bool_))
    structure = [tf.TensorSpec(shape=(3,), dtype=tf.bool)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    variant = tf.constant(np.array([1,2], dtype=np.int64))
    structure = [tf.TensorSpec(shape=(2,), dtype=tf.int64)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    variant = tf.constant(np.array(1, dtype=np.int64))
    structure = [tf.TensorSpec(shape=(), dtype=tf.int64)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    variant = tf.constant(np.array([[-1, -2],[-3, -4]], dtype=np.int32))
    structure = [tf.TensorSpec(shape=(2, 2), dtype=tf.int32)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    variant = tf.constant(np.array([1.1, 2.2, 3.3], dtype=np.float64))
    structure = [tf.TensorSpec(shape=(3,), dtype=tf.float64)]
    input_dict = {"variant": variant, "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.from_variant"] = tf_data_experimental_from_variant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.from_variant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.from_variant'.")

check_valid('tf.data.experimental.from_variant', generated_inputs['tf.data.experimental.from_variant'], lib="tf", suffix=0)
