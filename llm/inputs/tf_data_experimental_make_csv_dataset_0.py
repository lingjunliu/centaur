
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import os
import tempfile
import gzip

def tf_data_experimental_make_csv_dataset_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.make_csv_dataset function.
    This version avoids using arguments that are lists of strings (like column_names) or other
    non-numeric lists that might cause issues with the test harness.
    All inputs will use header inference (header=True).
    """
    
    temp_dir = tempfile.mkdtemp()
    
    file_path1 = os.path.join(temp_dir, 'data1.csv')
    with open(file_path1, 'w') as f:
        f.write("col1,col2,col3,label\n")
        f.write("1,1.1,a,0\n")
        f.write("2,2.2,b,1\n")
        f.write("3,3.3,c,0\n")

    file_path2 = os.path.join(temp_dir, 'data2.csv')
    with open(file_path2, 'w') as f:
        f.write("feature_a,feature_b,target\n")
        f.write("4,4.4,true\n")
        f.write("5,5.5,false\n")
        f.write("6,6.6,true\n")

    file_path_semi = os.path.join(temp_dir, 'data_semi.csv')
    with open(file_path_semi, 'w') as f:
        f.write("f1;f2;f3\n")
        f.write("10;10.1;x\n")
        f.write("20;20.2;y\n")

    file_path_na = os.path.join(temp_dir, 'data_na.csv')
    with open(file_path_na, 'w') as f:
        f.write("colA,colB,colC\n")
        f.write("1,,foo\n")
        f.write("2,2.2,MISSING\n")
        f.write("3,3.3,bar\n")
        
    file_path_gz = os.path.join(temp_dir, 'data.csv.gz')
    with gzip.open(file_path_gz, 'wt', encoding='utf-8') as f_gz:
        f_gz.write("g1,g2,g3\n")
        f_gz.write("1,2,3\n")
        f_gz.write("4,5,6\n")

    list_of_inputs = []
    
    base_input = {
        'file_pattern': None, 'batch_size': None, 'column_names': None, 
        'column_defaults': None, 'label_name': None, 'select_columns': None,
        'field_delim': ',', 'use_quote_delim': True, 'na_value': '', 
        'header': True, 'num_epochs': None, 'shuffle': True, 
        'shuffle_buffer_size': 10000, 'shuffle_seed': None, 
        'prefetch_buffer_size': None, 'num_parallel_reads': None, 'sloppy': False,
        'num_rows_for_inference': 100, 'compression_type': None, 
        'ignore_errors': False, 'encoding': 'utf-8'
    }

    # Input 1: Basic case, inferring from header.
    input_1 = copy.deepcopy(base_input)
    input_1.update({
        'file_pattern': [file_path1],
        'batch_size': 2,
    })
    list_of_inputs.append(input_1)

    # Input 2: Specify a label column.
    input_2 = copy.deepcopy(base_input)
    input_2.update({
        'file_pattern': [file_path1],
        'batch_size': 1,
        'label_name': 'label'
    })
    list_of_inputs.append(input_2)

    # Input 3: Select columns by integer index.
    input_3 = copy.deepcopy(base_input)
    input_3.update({
        'file_pattern': [file_path1],
        'batch_size': 1,
        'label_name': 'label',
        'select_columns': [0, 1, 3]
    })
    list_of_inputs.append(input_3)
    
    # Input 4: Use a different field delimiter.
    input_4 = copy.deepcopy(base_input)
    input_4.update({
        'file_pattern': [file_path_semi],
        'batch_size': 2,
        'field_delim': ';'
    })
    list_of_inputs.append(input_4)

    # Input 5: Disable shuffle and set a specific number of epochs.
    input_5 = copy.deepcopy(base_input)
    input_5.update({
        'file_pattern': [file_path2],
        'batch_size': 1,
        'num_epochs': 1,
        'shuffle': False
    })
    list_of_inputs.append(input_5)

    # Input 6: Multiple files in pattern and ignore errors.
    input_6 = copy.deepcopy(base_input)
    input_6.update({
        'file_pattern': [file_path1, file_path2],
        'batch_size': 4,
        'ignore_errors': True
    })
    list_of_inputs.append(input_6)

    # Input 7: Gzip compressed file.
    input_7 = copy.deepcopy(base_input)
    input_7.update({
        'file_pattern': [file_path_gz],
        'batch_size': 2,
        'compression_type': 'GZIP'
    })
    list_of_inputs.append(input_7)

    # Input 8: Advanced performance options.
    input_8 = copy.deepcopy(base_input)
    input_8.update({
        'file_pattern': [file_path1],
        'batch_size': 2,
        'shuffle_buffer_size': 50,
        'shuffle_seed': 42,
        'prefetch_buffer_size': 2,
        'num_parallel_reads': 2,
        'sloppy': True
    })
    list_of_inputs.append(input_8)

    # Input 9: Custom NA value.
    input_9 = copy.deepcopy(base_input)
    input_9.update({
        'file_pattern': [file_path_na],
        'batch_size': 3,
        'na_value': 'MISSING'
    })
    list_of_inputs.append(input_9)
    
    # Input 10: Another simple case with different batch size and file.
    input_10 = copy.deepcopy(base_input)
    input_10.update({
        'file_pattern': [file_path2],
        'batch_size': 3,
        'label_name': 'target'
    })
    list_of_inputs.append(input_10)

    return list_of_inputs

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

check_valid('tf.data.experimental.make_csv_dataset', generated_inputs['tf.data.experimental.make_csv_dataset'], lib="tf", suffix=0)
