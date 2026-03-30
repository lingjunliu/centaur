
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_make_csv_dataset_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'file_pattern': ['test.csv'],
        'batch_size': 32,
        'column_names': ['col1', 'col2', 'col3'],
        'column_defaults': [tf.constant(0.0, dtype=tf.float32), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32)],
        'label_name': 'col3',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': None,
        'num_parallel_reads': None,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'file_pattern': ['test1.csv', 'test2.csv'],
        'batch_size': 64,
        'column_names': ['feature1', 'feature2', 'target'],
        'column_defaults': [tf.constant(0.0, dtype=tf.float64), tf.constant('', dtype=tf.string), tf.constant(0, dtype=tf.int64)],
        'label_name': 'target',
        'select_columns': None,
        'field_delim': ';',
        'use_quote_delim': False,
        'na_value': 'NA',
        'header': False,
        'num_epochs': None,
        'shuffle': False,
        'shuffle_buffer_size': 5000,
        'shuffle_seed': None,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 4,
        'sloppy': True,
        'num_rows_for_inference': 50,
        'compression_type': 'GZIP',
        'ignore_errors': True,
        'encoding': 'latin1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'file_pattern': ['/path/to/data.csv'],
        'batch_size': 16,
        'column_names': ['a', 'b', 'c', 'd'],
        'column_defaults': [tf.constant(1, dtype=tf.int32), tf.constant(1.0, dtype=tf.float32), tf.constant("text", dtype=tf.string), tf.constant(True, dtype=tf.bool)],
        'label_name': 'd',
        'select_columns': [0,1,2,3],
        'field_delim': '|',
        'use_quote_delim': True,
        'na_value': 'None',
        'header': True,
        'num_epochs': 5,
        'shuffle': True,
        'shuffle_buffer_size': 10000,
        'shuffle_seed': 7,
        'prefetch_buffer_size': 100,
        'num_parallel_reads': 8,
        'sloppy': True,
        'num_rows_for_inference': 100,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'ascii'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'file_pattern': ['file.csv'],
        'batch_size': 256,
        'column_names': ['x', 'y'],
        'column_defaults': [tf.constant(1, dtype=tf.int32), tf.constant("default", dtype=tf.string)],
        'label_name': 'y',
        'select_columns': [0, 1],
        'field_delim': ',',
        'use_quote_delim': False,
        'na_value': '?',
        'header': False,
        'num_epochs': 10,
        'shuffle': False,
        'shuffle_buffer_size': 500,
        'shuffle_seed': 99,
        'prefetch_buffer_size': None,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 5,
        'compression_type': '',
        'ignore_errors': True,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'file_pattern': ['test.csv'],
        'batch_size': 32,
        'column_names': ['col1', 'col2', 'col3', 'col4', 'col5'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("text", dtype=tf.string), tf.constant(1, dtype=tf.int32), tf.constant(1.0, dtype=tf.float64), tf.constant(True, dtype=tf.bool)],
        'label_name': 'col3',
        'select_columns': [0, 1, 2, 3, 4],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': None,
        'num_parallel_reads': None,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'file_pattern': ['test.csv'],
        'batch_size': 32,
        'column_names': ['col1', 'col2', 'col3'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("text", dtype=tf.string), tf.constant(1, dtype=tf.int32)],
        'label_name': 'col3',
        'select_columns': [0, 1],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': None,
        'num_parallel_reads': None,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    input_dict = {
        'file_pattern': ['test.csv'],
        'batch_size': 32,
        'column_names': ['col1', 'col2', 'col3'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("text", dtype=tf.string), tf.constant(1, dtype=tf.int32)],
        'label_name': 'col3',
        'select_columns': [1,2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': None,
        'num_parallel_reads': None,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    input_dict = {
        'file_pattern': ['test.csv'],
        'batch_size': 32,
        'column_names': ['col1', 'col2', 'col3'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("text", dtype=tf.string), tf.constant(1, dtype=tf.int32)],
        'label_name': 'col3',
        'select_columns': ['col1', 'col2', 'col3'],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': None,
        'num_parallel_reads': None,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 11
    input_dict = {
        'file_pattern': ['data.csv'],
        'batch_size': 10,
        'column_names': ['feature1', 'feature2'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant('missing', dtype=tf.string)],
        'label_name': 'feature2',
        'select_columns': [0, 1],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'N/A',
        'header': True,
        'num_epochs': 2,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': None,
        'num_parallel_reads': None,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.make_csv_dataset"] = tf_data_experimental_make_csv_dataset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.make_csv_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.make_csv_dataset'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.make_csv_dataset', generated_inputs['tf.data.experimental.make_csv_dataset'], lib="tf", suffix=0)
