
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tempfile
import os
import atexit
import tensorflow as tf

# Create a temporary directory that will be cleaned up automatically.
temp_dir_obj = tempfile.TemporaryDirectory()
temp_dir = temp_dir_obj.name
atexit.register(temp_dir_obj.cleanup)

def create_tfrecord(filepath, compression_type="", records=None):
    """Creates a TFRecord file with optional compression."""
    if records is None:
        records = [b'record1', b'record2']
    dir_name = os.path.dirname(filepath)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    options = tf.io.TFRecordOptions(compression_type=compression_type if compression_type else None)
    with tf.io.TFRecordWriter(filepath, options=options) as writer:
        for record in records:
            writer.write(record)

# Create dummy TFRecord files with appropriate compression.
# These files will be found by the file_pattern argument.
create_tfrecord(os.path.join(temp_dir, 'train-01.tfrecord'))
create_tfrecord(os.path.join(temp_dir, 'train-02.tfrecord'))
create_tfrecord(os.path.join(temp_dir, 'file-aa.gz'), compression_type="GZIP")
create_tfrecord(os.path.join(temp_dir, 'file-ab.gz'), compression_type="GZIP")
create_tfrecord(os.path.join(temp_dir, 'shard_1.zlib'), compression_type="ZLIB")
create_tfrecord(os.path.join(temp_dir, 'shard_2.zlib'), compression_type="ZLIB")
create_tfrecord(os.path.join(temp_dir, 'a.record'))
create_tfrecord(os.path.join(temp_dir, 'test_data', 'data1'))
create_tfrecord(os.path.join(temp_dir, 'test_data', 'data2'))
create_tfrecord(os.path.join(temp_dir, 'part-00000'))
create_tfrecord(os.path.join(temp_dir, 'part-00001'))
create_tfrecord(os.path.join(temp_dir, 'dataset', 'files', 'data.tfrec'), compression_type="GZIP")
create_tfrecord(os.path.join(temp_dir, 'data_shards', 'shard_x'))
create_tfrecord(os.path.join(temp_dir, 'logs', 'log_a.log'))


def tf_raw_ops_record_input_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.RecordInput function.
    """
    list_of_inputs = []

    # Input 1: Basic usage with a glob pattern matching uncompressed files.
    input_dict_1 = {
        'file_pattern': os.path.join(temp_dir, 'train-*.tfrecord'),
        'file_random_seed': 301,
        'file_shuffle_shift_ratio': 0.0,
        'file_buffer_size': 10,
        'file_parallelism': 2,
        'batch_size': 2,
        'compression_type': '',
        'name': 'input_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: GZIP compression.
    input_dict_2 = {
        'file_pattern': os.path.join(temp_dir, 'file-??.gz'),
        'file_random_seed': 12345,
        'file_shuffle_shift_ratio': 0.0,
        'file_buffer_size': 5,
        'file_parallelism': 1,
        'batch_size': 1,
        'compression_type': 'GZIP',
        'name': 'gzip_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: ZLIB compression and a file shuffle shift.
    input_dict_3 = {
        'file_pattern': os.path.join(temp_dir, 'shard_*.zlib'),
        'file_random_seed': 42,
        'file_shuffle_shift_ratio': 0.5,
        'file_buffer_size': 20,
        'file_parallelism': 2,
        'batch_size': 2,
        'compression_type': 'ZLIB',
        'name': 'zlib_shuffled_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Minimal valid values, single uncompressed file.
    input_dict_4 = {
        'file_pattern': os.path.join(temp_dir, 'a.record'),
        'file_random_seed': 0,
        'file_shuffle_shift_ratio': 0.0,
        'file_buffer_size': 1,
        'file_parallelism': 1,
        'batch_size': 1,
        'compression_type': '',
        'name': 'minimal_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Negative random seed and subdirectory pattern.
    input_dict_5 = {
        'file_pattern': os.path.join(temp_dir, 'test_data', '*'),
        'file_random_seed': -100,
        'file_shuffle_shift_ratio': 0.0,
        'file_buffer_size': 10,
        'file_parallelism': 1,
        'batch_size': 2,
        'compression_type': '',
        'name': 'negative_seed_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Larger buffer and batch size.
    input_dict_6 = {
        'file_pattern': os.path.join(temp_dir, 'part-*'),
        'file_random_seed': 2023,
        'file_shuffle_shift_ratio': 0.0,
        'file_buffer_size': 10,
        'file_parallelism': 2,
        'batch_size': 2,
        'compression_type': '',
        'name': 'large_values_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Full shuffle shift ratio with GZIP compressed file.
    input_dict_7 = {
        'file_pattern': os.path.join(temp_dir, 'dataset', 'files', '*.tfrec'),
        'file_random_seed': 9876,
        'file_shuffle_shift_ratio': 1.0,
        'file_buffer_size': 10,
        'file_parallelism': 1,
        'batch_size': 1,
        'compression_type': 'GZIP',
        'name': 'full_shift_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Another valid float for shuffle shift ratio.
    input_dict_8 = {
        'file_pattern': os.path.join(temp_dir, 'data_shards', 'shard_*'),
        'file_random_seed': 555,
        'file_shuffle_shift_ratio': 0.25,
        'file_buffer_size': 15,
        'file_parallelism': 1,
        'batch_size': 1,
        'compression_type': '',
        'name': 'quarter_shift_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty string for name.
    input_dict_9 = {
        'file_pattern': os.path.join(temp_dir, 'logs', '*.log'),
        'file_random_seed': 301,
        'file_shuffle_shift_ratio': 0.0,
        'file_buffer_size': 10,
        'file_parallelism': 1,
        'batch_size': 1,
        'compression_type': '',
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Parallelism greater than number of files.
    input_dict_10 = {
        'file_pattern': os.path.join(temp_dir, 'train-*.tfrecord'), # Matches 2 files
        'file_random_seed': 7,
        'file_shuffle_shift_ratio': 0.0,
        'file_buffer_size': 10,
        'file_parallelism': 4,
        'batch_size': 2,
        'compression_type': '',
        'name': 'high_parallelism'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.RecordInput"] = tf_raw_ops_record_input_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RecordInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RecordInput'.")

check_valid('tf.raw_ops.RecordInput', generated_inputs['tf.raw_ops.RecordInput'], lib="tf", suffix=0)
