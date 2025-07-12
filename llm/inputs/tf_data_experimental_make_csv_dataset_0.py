
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
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32)],
        'label_name': 'col3',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'NA',
        'header': True,
        'num_epochs': 10,
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
        'column_names': ['feature1', 'feature2', 'target'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float64), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int64)],
        'label_name': 'target',
        'select_columns': [0, 1, 2],
        'field_delim': ';',
        'use_quote_delim': False,
        'na_value': '',
        'header': False,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 5000,
        'shuffle_seed': 123,
        'prefetch_buffer_size': 100,
        'num_parallel_reads': 4,
        'sloppy': True,
        'num_rows_for_inference': 50,
        'compression_type': 'GZIP',
        'ignore_errors': True,
        'encoding': 'latin-1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'file_pattern': ['test.csv'],
        'batch_size': 128,
        'column_names': ['f1', 'f2', 'f3', 'label'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)],
        'label_name': 'label',
        'select_columns': [0, 1, 2, 3],
        'field_delim': '\t',
        'use_quote_delim': True,
        'na_value': 'unknown',
        'header': True,
        'num_epochs': None,
        'shuffle': True,
        'shuffle_buffer_size': 20000,
        'shuffle_seed': None,
        'prefetch_buffer_size': None,
        'num_parallel_reads': None,
        'sloppy': False,
        'num_rows_for_inference': None,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'file_pattern': ['large_data.csv'],
        'batch_size': 256,
        'column_names': ['col_a', 'col_b', 'col_c', 'col_d', 'target_col'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float64), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int64), tf.constant(1.0, dtype=tf.float64), tf.constant(0, dtype=tf.int32)],
        'label_name': 'target_col',
        'select_columns': [0, 1, 2, 3, 4],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'N/A',
        'header': True,
        'num_epochs': 2,
        'shuffle': True,
        'shuffle_buffer_size': 15000,
        'shuffle_seed': 99,
        'prefetch_buffer_size': 500,
        'num_parallel_reads': 8,
        'sloppy': True,
        'num_rows_for_inference': 200,
        'compression_type': 'ZLIB',
        'ignore_errors': True,
        'encoding': 'ascii'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'file_pattern': ['data.csv'],
        'batch_size': 16,
        'column_names': ['feat1', 'feat2', 'label'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32)],
        'label_name': 'label',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '?',
        'header': True,
        'num_epochs': 5,
        'shuffle': True,
        'shuffle_buffer_size': 8000,
        'shuffle_seed': 77,
        'prefetch_buffer_size': 200,
        'num_parallel_reads': 2,
        'sloppy': False,
        'num_rows_for_inference': 75,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'file_pattern': ['abc.csv'],
        'batch_size': 4,
        'column_names': ['a', 'b', 'c'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32)],
        'label_name': 'c',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'NA',
        'header': True,
        'num_epochs': 1,
        'shuffle': False,
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

    # Input 7: Different delimiter and quote settings
    input_dict = {
        'file_pattern': ['path/to/file.dat'],
        'batch_size': 8,
        'column_names': ['col_A', 'col_B', 'label_col'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32)],
        'label_name': 'label_col',
        'select_columns': [0, 1, 2],
        'field_delim': '|',
        'use_quote_delim': False,
        'na_value': 'missing',
        'header': False,
        'num_epochs': 3,
        'shuffle': True,
        'shuffle_buffer_size': 500,
        'shuffle_seed': 1234,
        'prefetch_buffer_size': 50,
        'num_parallel_reads': 1,
        'sloppy': True,
        'num_rows_for_inference': 5,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using select_columns with column names
    input_dict = {
        'file_pattern': ['another_file.txt'],
        'batch_size': 10,
        'column_names': ['id', 'value1', 'value2', 'class'],
        'column_defaults': [tf.constant(0, dtype=tf.int32), tf.constant(1.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32), tf.constant("default", dtype=tf.string)],
        'label_name': 'class',
        'select_columns': [0, 1, 3],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'None',
        'header': True,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 100,
        'shuffle_seed': 5678,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': 4,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  compression and encoding settings
    input_dict = {
        'file_pattern': ['compressed_data.csv.gz'],
        'batch_size': 64,
        'column_names': ['col1', 'col2', 'col3'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float32), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32)],
        'label_name': 'col3',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'NA',
        'header': True,
        'num_epochs': 10,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': tf.data.AUTOTUNE,
        'num_parallel_reads': tf.data.AUTOTUNE,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': 'GZIP',
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10. remove file pattern
    input_dict = {
        'file_pattern': [],
        'batch_size': 64,
        'column_names': ['feature1', 'feature2', 'target'],
        'column_defaults': [tf.constant(1.0, dtype=tf.float64), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int64)],
        'label_name': 'target',
        'select_columns': [0, 1, 2],
        'field_delim': ';',
        'use_quote_delim': False,
        'na_value': '',
        'header': False,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 5000,
        'shuffle_seed': 123,
        'prefetch_buffer_size': 100,
        'num_parallel_reads': 4,
        'sloppy': True,
        'num_rows_for_inference': 50,
        'compression_type': 'GZIP',
        'ignore_errors': True,
        'encoding': 'latin-1'
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
