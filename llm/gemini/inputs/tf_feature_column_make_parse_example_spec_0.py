
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy


def tf_feature_column_make_parse_example_spec_inputs():
    """
    Generates a list of valid inputs for tf.feature_column.make_parse_example_spec.
    This version creates inputs where the 'feature_columns' list contains at most
    one element, to avoid triggering validation errors in external scripts that
    cannot handle lists of heterogeneous complex objects.
    """
    list_of_inputs = []

    # Input 1: An empty list of feature columns. This is a valid edge case.
    list_of_inputs.append({'feature_columns': []})

    # --- Inputs with a single FeatureColumn in the list ---

    # Input 2: A list with a single, basic numeric column.
    list_of_inputs.append({'feature_columns': [tf.feature_column.numeric_column('price')]})

    # Input 3: A list with a single numeric column with a scalar default value.
    list_of_inputs.append({'feature_columns': [tf.feature_column.numeric_column('score', default_value=-1.0)]})

    # Input 4: A list with a single numeric column with a non-default dtype.
    list_of_inputs.append({'feature_columns': [tf.feature_column.numeric_column('items_in_cart', dtype=tf.int64)]})

    # Input 5: A list with a single categorical column using a hash bucket.
    list_of_inputs.append({'feature_columns': [tf.feature_column.categorical_column_with_hash_bucket('product_id', hash_bucket_size=1000)]})

    # Input 6: A list with a single categorical column using identity.
    list_of_inputs.append({'feature_columns': [tf.feature_column.categorical_column_with_identity('class_id', num_buckets=5)]})

    # Input 7: A list with a single indicator column.
    cat_col_7 = tf.feature_column.categorical_column_with_hash_bucket('department', hash_bucket_size=20)
    list_of_inputs.append({'feature_columns': [tf.feature_column.indicator_column(cat_col_7)]})

    # Input 8: A list with a single embedding column.
    cat_col_8 = tf.feature_column.categorical_column_with_hash_bucket('user_id', hash_bucket_size=5000)
    list_of_inputs.append({'feature_columns': [tf.feature_column.embedding_column(cat_col_8, dimension=8)]})

    # Input 9: A list with a single indicator column wrapping a categorical_column_with_identity.
    cat_col_9 = tf.feature_column.categorical_column_with_identity('day_of_week', num_buckets=7)
    list_of_inputs.append({'feature_columns': [tf.feature_column.indicator_column(cat_col_9)]})

    # Input 10: A list with a single embedding column with a different dimension.
    cat_col_10 = tf.feature_column.categorical_column_with_identity('country_code', num_buckets=250)
    list_of_inputs.append({'feature_columns': [tf.feature_column.embedding_column(cat_col_10, dimension=16)]})

    return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.feature_column.make_parse_example_spec"] = tf_feature_column_make_parse_example_spec_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.make_parse_example_spec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.make_parse_example_spec'.")

check_valid('tf.feature_column.make_parse_example_spec', generated_inputs['tf.feature_column.make_parse_example_spec'], lib="tf", suffix=0)
