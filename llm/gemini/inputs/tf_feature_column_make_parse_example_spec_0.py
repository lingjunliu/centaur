
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
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single numeric column
    feature_columns = [tf.feature_column.numeric_column("feature_a")]
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple numeric columns
    feature_columns = [tf.feature_column.numeric_column("feature_a"), tf.feature_column.numeric_column("feature_b")]
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single categorical column with vocabulary list
    feature_columns = [tf.feature_column.categorical_column_with_vocabulary_list(key="feature_c", vocabulary_list=["a", "b", "c"])]
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Bucketized column
    feature_columns = [tf.feature_column.bucketized_column(tf.feature_column.numeric_column("feature_e"), boundaries=[0, 10, 20])]
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Indicator column
    feature_columns = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key="feature_f", vocabulary_list=["p", "q"]))]
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Numeric column with shape
    feature_columns = [tf.feature_column.numeric_column(key='feature_h', shape=(2,))]
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Numeric column with default value
    feature_columns = [tf.feature_column.numeric_column(key='feature_i', default_value=0.5)]
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Numeric column with dtype
    feature_columns = [tf.feature_column.numeric_column(key='feature_j', dtype=tf.int64)]
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Identity categorical column
    feature_columns = [tf.feature_column.categorical_column_with_identity(key='feature_k', num_buckets=10)]
    input_dict = {'feature_columns': feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.make_parse_example_spec', generated_inputs['tf.feature_column.make_parse_example_spec'], lib="tf", suffix=0)
