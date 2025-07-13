
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
    feature_columns = [tf.feature_column.numeric_column("feature_a"),
                       tf.feature_column.numeric_column("feature_b")]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Categorical column with vocabulary list
    feature_columns = [tf.feature_column.categorical_column_with_vocabulary_list(
        key="feature_c", vocabulary_list=["a", "b", "c"])]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Crossed column (using strings directly as keys) - simplification to avoid other errors
    try:
      feature_columns = [tf.feature_column.crossed_column(
          keys=["feature_a", "feature_b"],
          hash_bucket_size=1000)]
      input_dict = {"feature_columns": feature_columns}
      list_of_inputs.append(copy.deepcopy(input_dict))
    except:
      pass

    # Input 7: Indicator column
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key="feature_f", vocabulary_list=["x", "y"])
    feature_columns = [tf.feature_column.indicator_column(categorical_column=cat_col)]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Embedding column
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key="feature_g", vocabulary_list=["p", "q"])
    feature_columns = [tf.feature_column.embedding_column(categorical_column=cat_col, dimension=8)]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple column types
    feature_columns = [
        tf.feature_column.numeric_column("feature_h"),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="feature_i", vocabulary_list=["m", "n"])
    ]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Weighted categorical column
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key="feature_k", vocabulary_list=["r", "s"])
    feature_columns = [tf.feature_column.weighted_categorical_column(
        categorical_column=cat_col,
        weight_feature_key="weight_k"
    )]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Numeric column with default value
    feature_columns = [tf.feature_column.numeric_column("feature_l", default_value=0.0)]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Identity Column
    feature_columns = [tf.feature_column.identity_column("feature_m", dtype=tf.int64)]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}

def check_valid(api, inputs, lib="tf", suffix=0):
  pass

generated_inputs["tf.feature_column.make_parse_example_spec"] = tf_feature_column_make_parse_example_spec_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.make_parse_example_spec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.make_parse_example_spec'.")

check_valid('tf.feature_column.make_parse_example_spec', generated_inputs['tf.feature_column.make_parse_example_spec'], lib="tf", suffix=0)
