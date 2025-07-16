
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ArgMin_inputs():
    list_of_inputs = []

    # Input 1, valid
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dimension_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_1"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    dimension_tensor = np.array(1, dtype=np.int32)
    output_type = tf.int32
    name = "argmin_2"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    dimension_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_3"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, negative dimension
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    dimension_tensor = np.array(-1, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_4"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, float64
    input_tensor = np.array([1.0, 0.5, 3.0], dtype=np.float64)
    dimension_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_5"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, int64
    input_tensor = np.array([1, 2, 3], dtype=np.int64)
    dimension_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_6"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7, valid, 3D tensor
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dimension_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_7"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, different dimension
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dimension_tensor = np.array(1, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_8"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, dimension=2
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dimension_tensor = np.array(2, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_9"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, uint8
    input_tensor = np.array([1, 2, 0], dtype=np.uint8)
    dimension_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_10"

    input_dict = {
        "input": input_tensor,
        "dimension": dimension_tensor,
        "output_type": output_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ArgMin"] = tf_raw_ops_ArgMin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ArgMin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ArgMin'.")

check_valid('tf.raw_ops.ArgMin', generated_inputs['tf.raw_ops.ArgMin'], lib="tf", suffix=0)
