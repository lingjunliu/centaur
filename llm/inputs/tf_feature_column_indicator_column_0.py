
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_feature_column_indicator_column_inputs():
    list_of_inputs = []

    # Helper function to create categorical columns
    def create_categorical_column(vocabulary_list, column_name="test_column", dtype=tf.string, default_value=None):
        return tf.feature_column.categorical_column_with_vocabulary_list(
            key=column_name,
            vocabulary_list=vocabulary_list,
            dtype=dtype,
            default_value=default_value
        )

    # Input 1: Simple string vocabulary
    categorical_column = create_categorical_column(['a', 'b', 'c'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer vocabulary
    categorical_column = create_categorical_column([1, 2, 3], column_name="int_column", dtype=tf.int64)
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vocabulary with numbers represented as strings
    categorical_column = create_categorical_column(['1', '2', '3'], column_name="num_str_column")
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Vocabulary with a default value
    categorical_column = create_categorical_column(['x', 'y', 'z'], default_value=-1)
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger vocabulary
    categorical_column = create_categorical_column(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Vocabulary with special characters
    categorical_column = create_categorical_column(['!', '@', '#', '$'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer vocabulary with int32
    categorical_column = create_categorical_column([1, 2, 3], column_name="int32_column", dtype=tf.int32)
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Vocabulary with unicode characters
    categorical_column = create_categorical_column(['你好', '世界'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.indicator_column"] = tf_feature_column_indicator_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.indicator_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.indicator_column'.")

check_valid('tf.feature_column.indicator_column', generated_inputs['tf.feature_column.indicator_column'], lib="tf", suffix=0)
