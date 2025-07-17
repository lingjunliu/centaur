
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_feature_column_make_parse_example_spec_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    feature_columns = []
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single numeric column
    feature_columns = [tf.feature_column.numeric_column("feature_a")]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple numeric columns
    feature_columns = [
        tf.feature_column.numeric_column("feature_a"),
        tf.feature_column.numeric_column("feature_b"),
    ]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Categorical column with vocabulary list
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            "feature_c", vocabulary_list=["a", "b", "c"]
        )
    ]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Bucketized column
    feature_columns = [
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column("feature_d"), boundaries=[0, 1, 2]
        )
    ]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Crossed column (removed to avoid potential errors)

    # Input 7: Combination of different column types
    feature_columns = [
        tf.feature_column.numeric_column("feature_a"),
        tf.feature_column.categorical_column_with_vocabulary_list(
            "feature_b", vocabulary_list=["x", "y"]
        ),
    ]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Numeric column with default value
    feature_columns = [
        tf.feature_column.numeric_column("feature_h", default_value=0.0)
    ]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: indicator column
    feature_columns = [
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                "feature_f", vocabulary_list=["r", "s", "t"]
            )
        )
    ]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numeric column
    feature_columns = [tf.feature_column.numeric_column("feature_k")]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.make_parse_example_spec"] = tf_feature_column_make_parse_example_spec_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.make_parse_example_spec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.make_parse_example_spec'.")

check_valid('tf.feature_column.make_parse_example_spec', generated_inputs['tf.feature_column.make_parse_example_spec'], lib="tf", suffix=0)
