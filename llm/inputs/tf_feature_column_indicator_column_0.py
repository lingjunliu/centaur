
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_indicator_column_inputs():
    list_of_inputs = []

    # Input 1: categorical_column_with_vocabulary_list
    categorical_column1 = tf.feature_column.categorical_column_with_vocabulary_list(
        'colors', vocabulary_list=['red', 'green', 'blue'])
    input_dict1 = {'categorical_column': [categorical_column1]}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: categorical_column_with_identity
    categorical_column2 = tf.feature_column.categorical_column_with_identity(
        key='user_id', num_buckets=1000)
    input_dict2 = {'categorical_column': [categorical_column2]}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: categorical_column_with_hash_bucket
    categorical_column3 = tf.feature_column.categorical_column_with_hash_bucket(
        key='text', hash_bucket_size=100)
    input_dict3 = {'categorical_column': [categorical_column3]}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: crossed_column - Making sure vocabulary lists have uniform length for crossing.
    occupation_vocab = ['doctor', 'engineer', 'programmer', 'teacher', 'nurse']
    country_buckets = 5
    categorical_column_occupation = tf.feature_column.categorical_column_with_vocabulary_list(
        'occupation', vocabulary_list=occupation_vocab)
    categorical_column_country = tf.feature_column.categorical_column_with_identity(
        key='country_id', num_buckets=country_buckets)

    feature_columns = [categorical_column_occupation, categorical_column_country]
    crossed_column1 = tf.feature_column.crossed_column(
        feature_columns, hash_bucket_size=1000)
    input_dict4 = {'categorical_column': [crossed_column1]}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: categorical_column_with_vocabulary_file
    try:
        with open("city_vocabulary.txt", "w") as f:
            f.write("london\nparis\ntokyo\nnew_york\nberlin")
        categorical_column5 = tf.feature_column.categorical_column_with_vocabulary_file(
            key='city', vocabulary_file='city_vocabulary.txt', vocabulary_size=5)
        input_dict5 = {'categorical_column': [categorical_column5]}
        list_of_inputs.append(copy.deepcopy(input_dict5))
    except:
        pass

    # Input 6: A list of CategoricalColumns
    categorical_column6 = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', vocabulary_list=['sales', 'marketing', 'engineering'])
    categorical_column7 = tf.feature_column.categorical_column_with_identity(
        key='employee_id', num_buckets=500)
    input_dict6 = {'categorical_column': [categorical_column6, categorical_column7]}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: crossed_column - Making sure vocabulary lists have uniform length for crossing.
    product_vocab = ['A', 'B', 'C', 'D', 'E']
    customer_buckets = 10

    categorical_column_product = tf.feature_column.categorical_column_with_vocabulary_list(
        'product', vocabulary_list=product_vocab)
    categorical_column_customer = tf.feature_column.categorical_column_with_identity(
        key='customer_id', num_buckets=customer_buckets)

    feature_columns = [categorical_column_product, categorical_column_customer]
    crossed_column2 = tf.feature_column.crossed_column(
        feature_columns, hash_bucket_size=5000)
    input_dict7 = {'categorical_column': [crossed_column2]}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: categorical_column_with_hash_bucket with a smaller hash_bucket_size
    categorical_column8 = tf.feature_column.categorical_column_with_hash_bucket(
        key='keyword', hash_bucket_size=10)
    input_dict8 = {'categorical_column': [categorical_column8]}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: categorical_column_with_vocabulary_file
    try:
        with open("state_vocabulary.txt", "w") as f:
            f.write("california\ntexas\nflorida\nnew_york\n")
        categorical_column9 = tf.feature_column.categorical_column_with_vocabulary_file(
            key='state', vocabulary_file='state_vocabulary.txt', vocabulary_size=5)
        input_dict9 = {'categorical_column': [categorical_column9]}
        list_of_inputs.append(copy.deepcopy(input_dict9))
    except:
        pass

    # Input 10: Another categorical_column_with_identity
    categorical_column10 = tf.feature_column.categorical_column_with_identity(
        key='zip_code', num_buckets=200)
    input_dict10 = {'categorical_column': [categorical_column10]}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
