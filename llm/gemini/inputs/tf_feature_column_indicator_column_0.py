
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_feature_column_indicator_column_inputs():
    list_of_inputs = []

    # Helper function to create categorical columns (needed as input)
    def create_categorical_column(vocabulary_list, column_name="test_column"):
        return tf.feature_column.categorical_column_with_vocabulary_list(
            key=column_name, vocabulary_list=vocabulary_list, dtype=tf.string)

    # Input 1: Simple vocabulary list
    categorical_column = create_categorical_column(['a', 'b', 'c'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vocabulary with numbers
    categorical_column = create_categorical_column(['1', '2', '3'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vocabulary with special characters
    categorical_column = create_categorical_column(['!', '@', '#'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Vocabulary with mixed types
    categorical_column = create_categorical_column(['a', '1', '!'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  vocabulary list
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="empty_column", vocabulary_list=['valid'], dtype=tf.string)
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger vocabulary list
    categorical_column = create_categorical_column(['a', 'b', 'c', 'd', 'e', 'f'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Numerical vocabulary
    numerical_categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="numerical_column", vocabulary_list=[1, 2, 3], dtype=tf.int64
    )
    input_dict = {"categorical_column": [numerical_categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String vocabulary - Avoid numpy array and keep it consistent type
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="mixed_column", vocabulary_list=['1', 'b', '3'], dtype=tf.string
    )
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: No duplicates
    categorical_column = create_categorical_column(['x', 'y', 'z'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))
        
    # Input 10: Vocabulary list with empty string
    categorical_column = create_categorical_column(['', 'b', 'c'])
    input_dict = {"categorical_column": [categorical_column]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.indicator_column"] = tf_feature_column_indicator_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.indicator_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.indicator_column'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.indicator_column', generated_inputs['tf.feature_column.indicator_column'], lib="tf", suffix=0)
