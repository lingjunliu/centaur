
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
        'file_pattern': ['file1.csv', 'file2.csv'],
        'batch_size': 32,
        'column_names': ['col1', 'col2', 'col3'],
        'column_defaults': [tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([''], dtype=tf.string).numpy(), tf.constant([0], dtype=tf.int32).numpy()],
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
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': tf.data.AUTOTUNE,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'file_pattern': ['data*.csv'],
        'batch_size': 64,
        'column_names': None,
        'column_defaults': None,
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
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': 2,
        'sloppy': True,
        'num_rows_for_inference': 50,
        'compression_type': 'GZIP',
        'ignore_errors': True,
        'encoding': 'latin-1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'file_pattern': ['train.csv'],
        'batch_size': 128,
        'column_names': ['feature1', 'feature2', 'label'],
        'column_defaults': [tf.constant([0.0], dtype=tf.float64).numpy(), tf.constant([''], dtype=tf.string).numpy(), tf.constant([0], dtype=tf.int64).numpy()],
        'label_name': 'label',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '?',
        'header': True,
        'num_epochs': 5,
        'shuffle': True,
        'shuffle_buffer_size': 20000,
        'shuffle_seed': 123,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 4,
        'sloppy': False,
        'num_rows_for_inference': 200,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'file_pattern': ['test.csv'],
        'batch_size': 16,
        'column_names': ['id', 'x', 'y', 'class'],
        'column_defaults': [tf.constant([0], dtype=tf.int32).numpy(), tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([''], dtype=tf.string).numpy()],
        'label_name': 'class',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'None',
        'header': True,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': None,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'file_pattern': ['validation.csv'],
        'batch_size': 256,
        'column_names': None,
        'column_defaults': None,
        'label_name': 'outcome',
        'select_columns': None,
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 2,
        'shuffle': True,
        'shuffle_buffer_size': 15000,
        'shuffle_seed': 7,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': tf.data.AUTOTUNE,
        'sloppy': True,
        'num_rows_for_inference': 75,
        'compression_type': 'ZLIB',
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    column_defaults_6 = [tf.constant([1.0], dtype=tf.float32).numpy(), tf.constant([""], dtype=tf.string).numpy(), tf.constant([0], dtype=tf.int32).numpy()]
    input_dict = {
        'file_pattern': ['file_with_defaults.csv'],
        'batch_size': 8,
        'column_names': ['col_a', 'col_b', 'col_c'],
        'column_defaults': column_defaults_6,
        'label_name': 'col_c',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': tf.data.AUTOTUNE,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'file_pattern': ['sparse_data.csv'],
        'batch_size': 32,
        'column_names': ['feature1', 'feature2', 'feature3', 'target'],
        'column_defaults': [tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([0], dtype=tf.int32).numpy()],
        'label_name': 'target',
        'select_columns': [0, 1],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'NaN',
        'header': True,
        'num_epochs': 3,
        'shuffle': True,
        'shuffle_buffer_size': 5000,
        'shuffle_seed': 10,
        'prefetch_buffer_size': 5,
        'num_parallel_reads': 2,
        'sloppy': True,
        'num_rows_for_inference': 50,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'file_pattern': ['no_header.csv'],
        'batch_size': 64,
        'column_names': ['col_a', 'col_b', 'col_c'],
        'column_defaults': [tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([''], dtype=tf.string).numpy(), tf.constant([0], dtype=tf.int32).numpy()],
        'label_name': 'col_b',
        'select_columns': [0, 1, 2],
        'field_delim': ';',
        'use_quote_delim': False,
        'na_value': 'N/A',
        'header': False,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 2000,
        'shuffle_seed': 73,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': 4,
        'sloppy': False,
        'num_rows_for_inference': 150,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        'file_pattern': ['large_file.csv'],
        'batch_size': 1024,
        'column_names': ['f1', 'f2', 'f3', 'target'],
        'column_defaults': [tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([''], dtype=tf.string).numpy(), tf.constant([0], dtype=tf.int32).numpy(), tf.constant([0], dtype=tf.int32).numpy()],
        'label_name': 'target',
        'select_columns': [0,1,2,3],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': None,
        'shuffle': True,
        'shuffle_buffer_size': 100000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': tf.data.AUTOTUNE,
        'sloppy': True,
        'num_rows_for_inference': 100,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'file_pattern': ['missing_values.csv'],
        'batch_size': 32,
        'column_names': ['x', 'y', 'z', 'label'],
        'column_defaults': [tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([0], dtype=tf.int32).numpy()],
        'label_name': 'label',
        'select_columns': [0, 1, 2, 3],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '?',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 10000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': tf.data.AUTOTUNE,
        'sloppy': False,
        'num_rows_for_inference': 100,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        'file_pattern': ['single_file.csv'],
        'batch_size': 1,
        'column_names': ['feature_a', 'feature_b', 'target'],
        'column_defaults': [tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([''], dtype=tf.string).numpy(), tf.constant([0], dtype=tf.int32).numpy()],
        'label_name': 'target',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 10,
        'shuffle_seed': 12,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 5,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: select_columns as string
    input_dict = {
        'file_pattern': ['string_cols.csv'],
        'batch_size': 32,
        'column_names': ['feature_a', 'feature_b', 'target'],
        'column_defaults': [tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([''], dtype=tf.string).numpy(), tf.constant([0], dtype=tf.int32).numpy()],
        'label_name': 'target',
        'select_columns': ['feature_a', 'feature_b'],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': tf.data.AUTOTUNE,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: compression type ZLIB
    input_dict = {
        'file_pattern': ['compressed_file.csv'],
        'batch_size': 32,
        'column_names': ['col1', 'col2', 'col3'],
        'column_defaults': [tf.constant([0.0], dtype=tf.float32).numpy(), tf.constant([''], dtype=tf.string).numpy(), tf.constant([0], dtype=tf.int32).numpy()],
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
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': tf.data.AUTOTUNE,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': 'ZLIB',
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
    
    print("Valid")

if 'tf.data.experimental.make_csv_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.make_csv_dataset'.")

check_valid('tf.data.experimental.make_csv_dataset', generated_inputs['tf.data.experimental.make_csv_dataset'], lib="tf", suffix=0)
