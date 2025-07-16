
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
        'file_pattern': ['./data.csv'],
        'batch_size': 32,
        'column_names': ['col1', 'col2', 'col3'],
        'column_defaults': [tf.float32, tf.constant("", dtype=tf.string), tf.int32],
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
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'file_pattern': ['./data1.csv', './data2.csv'],
        'batch_size': 64,
        'column_names': ['feature1', 'feature2'],
        'column_defaults': [tf.int64, tf.constant("", dtype=tf.string)],
        'label_name': None,
        'select_columns': [0, 1],
        'field_delim': ';',
        'use_quote_delim': False,
        'na_value': 'NA',
        'header': False,
        'num_epochs': 2,
        'shuffle': False,
        'shuffle_buffer_size': 5000,
        'shuffle_seed': None,
        'prefetch_buffer_size': 5,
        'num_parallel_reads': 4,
        'sloppy': True,
        'num_rows_for_inference': 50,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'ISO-8859-1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'file_pattern': ['./data3.csv'],
        'batch_size': 128,
        'column_names': ['a', 'b', 'c', 'd'],
        'column_defaults': [tf.float64, tf.int32, tf.constant("", dtype=tf.string), tf.float32],
        'label_name': 'd',
        'select_columns': None,
        'field_delim': '\t',
        'use_quote_delim': True,
        'na_value': 'null',
        'header': True,
        'num_epochs': None,
        'shuffle': True,
        'shuffle_buffer_size': 10000,
        'shuffle_seed': 123,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 2,
        'sloppy': False,
        'num_rows_for_inference': None,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'UTF-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'file_pattern': ['./data4.csv'],
        'batch_size': 16,
        'column_names': ['x', 'y', 'z'],
        'column_defaults': [tf.int32, tf.float32, tf.constant("", dtype=tf.string)],
        'label_name': 'z',
        'select_columns': [0, 1],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 100,
        'shuffle_seed': 7,
        'prefetch_buffer_size': 5,
        'num_parallel_reads': 1,
        'sloppy': True,
        'num_rows_for_inference': 5,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'file_pattern': ['./data5.csv'],
        'batch_size': 256,
        'column_names': ['col_a', 'col_b'],
        'column_defaults': [tf.float32, tf.int64],
        'label_name': None,
        'select_columns': None,
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 20000,
        'shuffle_seed': 99,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 100,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'file_pattern': ['./data6.csv'],
        'batch_size': 8,
        'column_names': ['f1', 'f2', 'f3'],
        'column_defaults': [tf.constant("", dtype=tf.string), tf.int32, tf.float64],
        'label_name': 'f2',
        'select_columns': [0, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'N/A',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 500,
        'shuffle_seed': 1,
        'prefetch_buffer_size': 5,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_dict = {
        'file_pattern': ['./data7.csv'],
        'batch_size': 4,
        'column_names': ['a', 'b', 'c'],
        'column_defaults': [tf.int32, tf.constant("", dtype=tf.string), tf.float32],
        'label_name': 'c',
        'select_columns': [0,1],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': 5,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'file_pattern': ['./data8.csv', './data9.csv'],
        'batch_size': 32,
        'column_names': ['feat1', 'feat2'],
        'column_defaults': [tf.int64, tf.constant("", dtype=tf.string)],
        'label_name': None,
        'select_columns': [0, 1],
        'field_delim': ';',
        'use_quote_delim': False,
        'na_value': 'NA',
        'header': False,
        'num_epochs': 2,
        'shuffle': False,
        'shuffle_buffer_size': 5000,
        'shuffle_seed': None,
        'prefetch_buffer_size': 5,
        'num_parallel_reads': 4,
        'sloppy': True,
        'num_rows_for_inference': 50,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'ISO-8859-1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'file_pattern': ['./data9.csv'],
        'batch_size': 16,
        'column_names': ['x', 'y', 'z'],
        'column_defaults': [tf.int32, tf.float32, tf.constant("", dtype=tf.string)],
        'label_name': 'z',
        'select_columns': [0, 1],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 100,
        'shuffle_seed': 7,
        'prefetch_buffer_size': 5,
        'num_parallel_reads': 1,
        'sloppy': True,
        'num_rows_for_inference': 5,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'file_pattern': ['./data10.csv'],
        'batch_size': 256,
        'column_names': ['col_a', 'col_b'],
        'column_defaults': [tf.float32, tf.int64],
        'label_name': None,
        'select_columns': None,
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 20000,
        'shuffle_seed': 99,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 100,
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
    
    print("Valid")

if 'tf.data.experimental.make_csv_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.make_csv_dataset'.")

check_valid('tf.data.experimental.make_csv_dataset', generated_inputs['tf.data.experimental.make_csv_dataset'], lib="tf", suffix=0)
