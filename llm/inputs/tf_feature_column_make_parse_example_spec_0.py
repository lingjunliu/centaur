
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os
import tempfile
import numpy as np

def generate_make_parse_example_spec_inputs():
    """
    Generates a list of valid inputs for tf.feature_column.make_parse_example_spec.
    This version uses tf.DType for dtype arguments to avoid AttributeErrors.
    """
    list_of_inputs = []

    # Helper to create a dummy vocab file
    def create_vocab_file():
        try:
            tmpdir = tempfile.mkdtemp()
            vocab_file = os.path.join(tmpdir, "vocab.txt")
            with open(vocab_file, "w") as f:
                f.write("apple\nbanana\norange\n")
            return vocab_file, tmpdir
        except Exception:
            return None, None

    vocab_file_path, temp_dir_path = create_vocab_file()

    # To avoid errors in the testing framework, most inputs will contain
    # a list with a single, self-contained feature column.

    # Input 1: Basic numeric column with default dtype (tf.float32)
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.numeric_column(key='price')
    ]})

    # Input 2: Numeric column with a shape and integer dtype
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.numeric_column(key='image', shape=(64, 64), dtype=tf.int64)
    ]})

    # Input 3: Numeric column with a default value and TF dtype (FIXED)
    # The previous error was caused by using np.float32 which lacks '.is_floating'
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.numeric_column(key='score', default_value=-1.0, dtype=tf.float32)
    ]})

    # Input 4: Categorical column with an inline vocabulary
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='color', vocabulary_list=['R', 'G', 'B'])
    ]})

    # Input 5: Categorical column with a hash bucket
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.categorical_column_with_hash_bucket(
            key='product_id', hash_bucket_size=1000)
    ]})

    # Input 6: Categorical column with identity, using TF dtype
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.categorical_column_with_identity(
            key='class_id', num_buckets=5, dtype=tf.int64)
    ]})

    # Input 7: Categorical column from a vocabulary file
    if vocab_file_path:
        list_of_inputs.append({'feature_columns': [
            tf.feature_column.categorical_column_with_vocabulary_file(
                key='fruit', vocabulary_file=vocab_file_path, num_oov_buckets=1)
        ]})

    # Input 8: Bucketized column
    price_numeric = tf.feature_column.numeric_column('price_for_bucket')
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.bucketized_column(price_numeric, boundaries=[10., 20., 30.])
    ]})

    # Input 9: Indicator column
    color_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        'indicator_color', vocabulary_list=['R', 'G', 'B'])
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.indicator_column(color_categorical)
    ]})

    # Input 10: Embedding column
    product_categorical = tf.feature_column.categorical_column_with_hash_bucket(
        'product_for_embedding', hash_bucket_size=50)
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.embedding_column(product_categorical, dimension=10)
    ]})
    
    # Input 11: Weighted categorical column
    tag_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        'tag', vocabulary_list=['news', 'sports', 'tech'])
    list_of_inputs.append({'feature_columns': [
        tf.feature_column.weighted_categorical_column(tag_categorical, 'tag_weight')
    ]})

    # Input 12: Empty list of feature columns
    list_of_inputs.append({'feature_columns': []})

    # Cleanup
    if temp_dir_path:
        try:
            for filename in os.listdir(temp_dir_path):
                os.remove(os.path.join(temp_dir_path, filename))
            os.rmdir(temp_dir_path)
        except (OSError, FileNotFoundError):
            pass

    return list_of_inputs

generated_inputs["tf.feature_column.make_parse_example_spec"] = generate_make_parse_example_spec_inputs()

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
