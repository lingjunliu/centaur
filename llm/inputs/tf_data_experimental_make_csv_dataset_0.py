
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_make_csv_dataset_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.make_csv_dataset function.
    The fix is to convert lists of strings to numpy arrays. This is to work around a bug in the
    testing environment that fails to compute np.min on a standard list of strings, causing a
    UFuncNoLoopError. Providing a numpy array directly might be handled correctly. This also
    aligns with the 'numpy format' suggestion in the prompt. Lists of integers or empty lists
    are left as is, as they do not cause this issue.
    """
    list_of_inputs = []

    # Input 1: Basic case with header inference and a label.
    input_1 = {
        'file_pattern': np.array(['./file1.csv']),
        'batch_size': 8,
        'column_names': [],
        'column_defaults': [],
        'label_name': 'target',
        'select_columns': [],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 42,
        'prefetch_buffer_size': 1,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 100,
        'compression_type': '',
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: No header, explicit column names provided.
    input_2 = {
        'file_pattern': np.array(['./file2.csv']),
        'batch_size': 4,
        'column_names': np.array(['age', 'sex', 'thal', 'target']),
        'column_defaults': [],
        'label_name': 'target',
        'select_columns': [],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '?',
        'header': False,
        'num_epochs': 1,
        'shuffle': False,
        'shuffle_buffer_size': 10000,
        'shuffle_seed': 0,
        'prefetch_buffer_size': 0,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 50,
        'compression_type': '',
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Select specific columns by index.
    input_3 = {
        'file_pattern': np.array(['./file3.csv']),
        'batch_size': 32,
        'column_names': [],
        'column_defaults': [],
        'label_name': 'species',
        'select_columns': [0, 2, 4],
        'header': True,
        'num_epochs': 5,
        'shuffle': True,
        'shuffle_buffer_size': 5000,
        'shuffle_seed': 123,
        'prefetch_buffer_size': 2,
        'num_parallel_reads': 2,
        'sloppy': True,
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'num_rows_for_inference': 200,
        'compression_type': '',
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Select columns by name, no header.
    input_4 = {
        'file_pattern': np.array(['./file4.csv']),
        'batch_size': 64,
        'column_names': np.array(['a', 'b', 'c', 'd', 'e']),
        'column_defaults': [],
        'label_name': 'e',
        'select_columns': np.array(['a', 'c', 'e']),
        'field_delim': ';',
        'use_quote_delim': False,
        'na_value': 'NA',
        'header': False,
        'num_epochs': 1,
        'shuffle': True,
        'shuffle_buffer_size': 10000,
        'shuffle_seed': 123,
        'prefetch_buffer_size': 4,
        'num_parallel_reads': 4,
        'sloppy': False,
        'num_rows_for_inference': 100,
        'compression_type': '',
        'ignore_errors': True,
        'encoding': 'latin-1'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Read a compressed file with performance tuning.
    input_5 = {
        'file_pattern': np.array(['./file5.csv.gz']),
        'batch_size': 128,
        'column_names': [],
        'column_defaults': [],
        'label_name': 'target',
        'select_columns': [],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': True,
        'num_epochs': 10,
        'shuffle': True,
        'shuffle_buffer_size': 20000,
        'shuffle_seed': 2023,
        'prefetch_buffer_size': 8,
        'num_parallel_reads': 8,
        'sloppy': True,
        'num_rows_for_inference': 1000,
        'compression_type': 'GZIP',
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: Use a glob pattern for multiple files.
    input_6 = {
        'file_pattern': np.array(['./data_part_*.csv']),
        'batch_size': 10,
        'column_names': [],
        'column_defaults': [],
        'label_name': 'col_5',
        'select_columns': [],
        'field_delim': '\t',
        'use_quote_delim': True,
        'na_value': 'null',
        'header': True,
        'num_epochs': 2,
        'shuffle': True,
        'shuffle_buffer_size': 1000,
        'shuffle_seed': 7,
        'prefetch_buffer_size': 1,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 100,
        'compression_type': '',
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: No shuffling, multiple epochs, no label.
    input_7 = {
        'file_pattern': np.array(['./fixed_order_data.csv']),
        'batch_size': 1,
        'column_names': np.array(['id', 'value', 'category']),
        'column_defaults': [],
        'label_name': '',
        'select_columns': [],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'header': False,
        'num_epochs': 3,
        'shuffle': False,
        'shuffle_buffer_size': 1,
        'shuffle_seed': 1,
        'prefetch_buffer_size': 0,
        'num_parallel_reads': 1,
        'sloppy': False,
        'num_rows_for_inference': 100,
        'compression_type': '',
        'ignore_errors': False,
        'encoding': 'utf-8'
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    return list_of_inputs

generated_inputs["tf.data.experimental.make_csv_dataset"] = generate_make_csv_dataset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.make_csv_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.make_csv_dataset'.")

check_valid('tf.data.experimental.make_csv_dataset', generated_inputs['tf.data.experimental.make_csv_dataset'], lib="tf", suffix=0)
