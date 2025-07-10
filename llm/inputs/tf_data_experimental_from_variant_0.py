
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_from_variant_inputs():
    list_of_inputs = []

    # Helper function to create a dummy tf.variant tensor
    def create_dummy_variant_tensor():
        # Create a tf.data.Dataset and convert it to a variant tensor
        dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        variant_tensor = tf.compat.v1.data.experimental.to_variant(dataset)
        return variant_tensor

    # Input 1, valid
    variant = create_dummy_variant_tensor()
    structure = [tf.TensorSpec(shape=(), dtype=tf.int32)]
    input_dict = {"variant": variant.numpy(), "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    variant = create_dummy_variant_tensor()
    structure = [tf.TensorSpec(shape=(None,), dtype=tf.float32)]
    input_dict = {"variant": variant.numpy(), "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    variant = create_dummy_variant_tensor()
    structure = [tf.TensorSpec(shape=(2, 3), dtype=tf.string)]
    input_dict = {"variant": variant.numpy(), "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    variant = create_dummy_variant_tensor()
    structure = [tf.TensorSpec(shape=(None, None, 5), dtype=tf.bool)]
    input_dict = {"variant": variant.numpy(), "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    variant = create_dummy_variant_tensor()
    structure = [tf.TensorSpec(shape=(1, 2, 3, 4), dtype=tf.int64)]
    input_dict = {"variant": variant.numpy(), "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, nested structure
    variant = create_dummy_variant_tensor()
    structure = [[tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(None,), dtype=tf.float32)]]
    input_dict = {"variant": variant.numpy(), "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, different dtypes
    variant = create_dummy_variant_tensor()
    structure = [tf.TensorSpec(shape=(), dtype=tf.uint8)]
    input_dict = {"variant": variant.numpy(), "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, different shapes
    variant = create_dummy_variant_tensor()
    structure = [tf.TensorSpec(shape=(1,), dtype=tf.int32)]
    input_dict = {"variant": variant.numpy(), "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, different rank
    variant = create_dummy_variant_tensor()
    structure = [tf.TensorSpec(shape=(1,1), dtype=tf.int32)]
    input_dict = {"variant": variant.numpy(), "structure": structure}
    list_of_inputs.append(copy.deepcopy(input_dict))
  
    # Input 10, valid, nested structure of size 3
    variant = create_dummy_variant_tensor()
    structure = [tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(None,), dtype=tf.float32), tf.TensorSpec(shape=(2, 3), dtype=tf.string)]
    input_dict = {"variant": variant.numpy(), "structure": structure}
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
