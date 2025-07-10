
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
        'column_defaults': [np.array(0.0, dtype=np.float32), np.array("", dtype=np.string_), np.array(0, dtype=np.int32)],
        'label_name': 'col3',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 10,
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
        'file_pattern': ['data.csv'],
        'batch_size': 64,
        'column_names': ['feature1', 'feature2', 'target'],
        'column_defaults': [np.array(0.0, dtype=np.float64), np.array("", dtype=np.string_), np.array(0, dtype=np.int64)],
        'label_name': 'target',
        'select_columns': [0, 1, 2],
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
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'latin-1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'file_pattern': ['train.csv', 'val.csv'],
        'batch_size': 128,
        'column_names': ['a', 'b', 'c', 'd'],
        'column_defaults': [np.array(0, dtype=np.int32), np.array("", dtype=np.string_), np.array(0.0, dtype=np.float32), np.array(0, dtype=np.int32)],
        'label_name': 'd',
        'select_columns': [0, 1, 2, 3],
        'field_delim': '\t',
        'use_quote_delim': True,
        'na_value': '?',
        'header': True,
        'num_epochs': 5,
        'shuffle': True,
        'shuffle_buffer_size': 2000,
        'shuffle_seed': 123,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 2,
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
        'column_names': ['x', 'y', 'z'],
        'column_defaults': [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)],
        'label_name': 'z',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'NaN',
        'header': True,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 100,
        'shuffle_seed': 7,
        'prefetch_buffer_size': 5,
        'num_parallel_reads': 1,
        'sloppy': True,
        'num_rows_for_inference': 1000,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'file_pattern': ['*.csv'],
        'batch_size': 256,
        'column_names': ['col_a', 'col_b', 'col_c'],
        'column_defaults': [np.array("", dtype=np.string_), np.array(0, dtype=np.int64), np.array(0.0, dtype=np.float64)],
        'label_name': 'col_c',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': False,
        'num_epochs': 2,
        'shuffle': True,
        'shuffle_buffer_size': 8000,
        'shuffle_seed': 99,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 8,
        'sloppy': False,
        'num_rows_for_inference': None,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_dict = {
        'file_pattern': ['data1.csv', 'data2.csv'],
        'batch_size': 8,
        'column_names': ['f1', 'f2', 'f3', 'l'],
        'column_defaults': [np.array(0.0, dtype=np.float32), np.array(0, dtype=np.int32), np.array("", dtype=np.string_), np.array(0, dtype=np.int32)],
        'label_name': 'l',
        'select_columns': [0, 1, 2, 3],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'N/A',
        'header': True,
        'num_epochs': 3,
        'shuffle': True,
        'shuffle_buffer_size': 1024,
        'shuffle_seed': 10,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 2,
        'sloppy': False,
        'num_rows_for_inference': 100,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_dict = {
        'file_pattern': ['single.csv'],
        'batch_size': 4,
        'column_names': ['feat1', 'feat2', 'lab'],
        'column_defaults': [np.array(0.0, dtype=np.float64), np.array("", dtype=np.string_), np.array(0, dtype=np.int64)],
        'label_name': 'lab',
        'select_columns': [0, 1, 2],
        'field_delim': ';',
        'use_quote_delim': False,
        'na_value': 'unknown',
        'header': False,
        'num_epochs': None,
        'shuffle': False,
        'shuffle_buffer_size': 2048,
        'shuffle_seed': None,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 1,
        'sloppy': True,
        'num_rows_for_inference': 50,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'latin1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'file_pattern': ['a.csv', 'b.csv', 'c.csv'],
        'batch_size': 32,
        'column_names': ['one', 'two', 'three', 'four'],
        'column_defaults': [np.array("", dtype=np.string_), np.array(0, dtype=np.int32), np.array(0.0, dtype=np.float32), np.array(0, dtype=np.int32)],
        'label_name': 'four',
        'select_columns': [0, 1, 2, 3],
        'field_delim': '\t',
        'use_quote_delim': True,
        'na_value': 'None',
        'header': True,
        'num_epochs': 2,
        'shuffle': True,
        'shuffle_buffer_size': 4096,
        'shuffle_seed': 42,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 4,
        'sloppy': False,
        'num_rows_for_inference': 20,
        'compression_type': None,
        'ignore_errors': True,
        'encoding': 'ascii'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'file_pattern': ['mydata.csv'],
        'batch_size': 64,
        'column_names': ['A', 'B', 'C'],
        'column_defaults': [np.array(0.0, dtype=np.float32), np.array(0, dtype=np.int64), np.array("", dtype=np.string_)],
        'label_name': 'C',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 128,
        'shuffle_seed': 7,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 1,
        'sloppy': True,
        'num_rows_for_inference': 10,
        'compression_type': None,
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'file_pattern': ['all_data.csv'],
        'batch_size': 128,
        'column_names': ['feature_1', 'feature_2', 'target_var'],
        'column_defaults': [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32), np.array(0, dtype=np.int32)],
        'label_name': 'target_var',
        'select_columns': [0, 1, 2],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'nan',
        'header': True,
        'num_epochs': 3,
        'shuffle': True,
        'shuffle_buffer_size': 256,
        'shuffle_seed': 1234,
        'prefetch_buffer_size': 10,
        'num_parallel_reads': 2,
        'sloppy': False,
        'num_rows_for_inference': 100,
        'compression_type': None,
        'ignore_errors': True,
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
