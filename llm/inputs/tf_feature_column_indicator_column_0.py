
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy


def tf_feature_column_indicator_column_inputs():
    """
    Generates a list of valid inputs for tf.feature_column.indicator_column.
    The 'categorical_column' must be a CategoricalColumn object for the input to be valid.
    This version avoids complex categorical columns like CrossedColumn to prevent
    downstream processing errors in tools that inspect object structure.
    """
    list_of_inputs = []

    # --- Inputs using categorical_column_with_vocabulary_list ---

    # Input 1: Basic string vocabulary
    cat_col_1 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='product_class',
        vocabulary_list=['kitchen', 'electronics', 'sports']
    )
    input_dict_1 = {'categorical_column': cat_col_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Integer vocabulary
    cat_col_2 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='product_id',
        vocabulary_list=[101, 102, 103, 104],
        dtype=tf.int64
    )
    input_dict_2 = {'categorical_column': cat_col_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With a default value for OOV items
    cat_col_3 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='department',
        vocabulary_list=['hr', 'eng', 'sales'],
        default_value=np.int64(-1)
    )
    input_dict_3 = {'categorical_column': cat_col_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With out-of-vocabulary (OOV) buckets
    cat_col_4 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='feature_with_oov',
        vocabulary_list=['A', 'B', 'C'],
        num_oov_buckets=np.int32(5)
    )
    input_dict_4 = {'categorical_column': cat_col_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Single item in vocabulary
    cat_col_5 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='singleton_feature',
        vocabulary_list=['unique_value']
    )
    input_dict_5 = {'categorical_column': cat_col_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # --- Inputs using categorical_column_with_hash_bucket ---

    # Input 6: Basic hash bucket with small size
    cat_col_6 = tf.feature_column.categorical_column_with_hash_bucket(
        key='user_id_hashed',
        hash_bucket_size=np.int32(10)
    )
    input_dict_6 = {'categorical_column': cat_col_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Hash bucket with large size
    cat_col_7 = tf.feature_column.categorical_column_with_hash_bucket(
        key='keywords_hashed',
        hash_bucket_size=np.int64(100000)
    )
    input_dict_7 = {'categorical_column': cat_col_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Hash bucket for integer features
    cat_col_8 = tf.feature_column.categorical_column_with_hash_bucket(
        key='int_feature_hashed',
        hash_bucket_size=50,
        dtype=tf.int64
    )
    input_dict_8 = {'categorical_column': cat_col_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # --- Input using categorical_column_with_identity ---
    # Input 9: Identity column. This replaces the invalid combination.
    cat_col_9 = tf.feature_column.categorical_column_with_identity(
        key='identity_feature',
        num_buckets=100
    )
    input_dict_9 = {'categorical_column': cat_col_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # --- Another hash bucket variant ---
    # Input 10: Hash bucket with size 1
    cat_col_10 = tf.feature_column.categorical_column_with_hash_bucket(
        key='single_hash_bucket',
        hash_bucket_size=1
    )
    input_dict_10 = {'categorical_column': cat_col_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

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

check_valid('tf.feature_column.indicator_column', generated_inputs['tf.feature_column.indicator_column'], lib="tf", suffix=0)
