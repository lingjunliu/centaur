
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_group_inputs():
    list_of_inputs = []

    def create_dummy_tensor(shape, dtype=tf.float32):
        return tf.constant(np.zeros(shape, dtype=np.float32 if dtype == tf.float32 else np.int32))

    # Input 1: Empty list of tensors
    inputs = []
    name = "empty_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single tensor
    inputs = [create_dummy_tensor((1,))]
    name = "single_tensor_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple tensors
    inputs = [create_dummy_tensor((1,)), create_dummy_tensor((2,))]
    name = "multiple_tensors_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensors with different shapes
    inputs = [create_dummy_tensor((1, 2)), create_dummy_tensor((2, 2))]
    name = "different_shapes_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensors with different dtypes
    inputs = [create_dummy_tensor((1,), dtype=tf.int32), create_dummy_tensor((1,), dtype=tf.float32)]
    name = "different_dtypes_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensors with name
    a = create_dummy_tensor((1,), name='a')
    b = create_dummy_tensor((2,), name='b')
    inputs = [a, b]
    name = "named_tensors_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher dimensional tensors
    inputs = [create_dummy_tensor((2, 3)), create_dummy_tensor((5, 2))]
    name = "high_dimensional_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values
    inputs = [tf.constant(-1.0), tf.constant([-2.0, -3.0])]
    name = "negative_values_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List Comprehension Generated Tensors
    inputs = [create_dummy_tensor((i,)) for i in range(1, 3)]
    name = "list_comprehension_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Simple Tensors
    inputs = [create_dummy_tensor((1,)), create_dummy_tensor((1,))]
    name = "simple_tensors_group"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.group"] = tf_group_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.group' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.group'.")

check_valid('tf.group', generated_inputs['tf.group'], lib="tf", suffix=0)
