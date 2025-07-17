
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os
import glob

def tf_raw_ops_record_input_inputs():
    list_of_inputs = []

    # Create a dummy tfrecords file
    def create_dummy_tfrecords(filename):
        try:
            with tf.io.TFRecordWriter(filename) as writer:
                for i in range(10):
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'i': tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
                    }))
                    writer.write(example.SerializeToString())
        except Exception as e:
            print(f"Error creating TFRecord file: {e}")

    # Input 1
    file_pattern = "./data.tfrecords"
    create_dummy_tfrecords(file_pattern)
    file_random_seed = 42
    file_shuffle_shift_ratio = 0.1
    file_buffer_size = 5000
    file_parallelism = 8
    batch_size = 16
    compression_type = ""
    name = "record_input_1"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    file_pattern = "./data_*.tfrecords"
    create_dummy_tfrecords("./data_1.tfrecords")
    file_random_seed = 123
    file_shuffle_shift_ratio = 0.5
    file_buffer_size = 1000
    file_parallelism = 4
    batch_size = 64
    compression_type = "ZLIB"
    name = "record_input_2"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    file_pattern = "./train_data.tfrecords"
    create_dummy_tfrecords(file_pattern)
    file_random_seed = 7
    file_shuffle_shift_ratio = 0.0
    file_buffer_size = 2000
    file_parallelism = 1
    batch_size = 32
    compression_type = "GZIP"
    name = "record_input_3"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    file_pattern = "./valid_*.tfrecords"
    create_dummy_tfrecords("./valid_1.tfrecords")
    file_random_seed = 99
    file_shuffle_shift_ratio = 1.0
    file_buffer_size = 15000
    file_parallelism = 32
    batch_size = 128
    compression_type = ""
    name = "record_input_4"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    file_pattern = "./test.tfrecords"
    create_dummy_tfrecords(file_pattern)
    file_random_seed = 0
    file_shuffle_shift_ratio = 0.25
    file_buffer_size = 500
    file_parallelism = 2
    batch_size = 8
    compression_type = "ZLIB"
    name = "record_input_5"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    file_pattern = "./data_set.tfrecords"
    create_dummy_tfrecords(file_pattern)
    file_random_seed = 1
    file_shuffle_shift_ratio = 0.75
    file_buffer_size = 7500
    file_parallelism = 10
    batch_size = 48
    compression_type = "GZIP"
    name = "record_input_6"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    file_pattern = "./final_data.tfrecords"
    create_dummy_tfrecords(file_pattern)
    file_random_seed = 2024
    file_shuffle_shift_ratio = 0.33
    file_buffer_size = 2500
    file_parallelism = 6
    batch_size = 24
    compression_type = ""
    name = "record_input_7"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    file_pattern = "./backup.tfrecords"
    create_dummy_tfrecords(file_pattern)
    file_random_seed = 66
    file_shuffle_shift_ratio = 0.66
    file_buffer_size = 12000
    file_parallelism = 20
    batch_size = 96
    compression_type = "ZLIB"
    name = "record_input_8"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    file_pattern = "./sample.tfrecords"
    create_dummy_tfrecords(file_pattern)
    file_random_seed = 444
    file_shuffle_shift_ratio = 0.99
    file_buffer_size = 100
    file_parallelism = 1
    batch_size = 2
    compression_type = "GZIP"
    name = "record_input_9"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    file_pattern = "./large_data.tfrecords"
    create_dummy_tfrecords(file_pattern)
    file_random_seed = 500
    file_shuffle_shift_ratio = 0.8
    file_buffer_size = 20000
    file_parallelism = 64
    batch_size = 256
    compression_type = ""
    name = "record_input_10"

    input_dict = {
        "file_pattern": file_pattern,
        "file_random_seed": file_random_seed,
        "file_shuffle_shift_ratio": file_shuffle_shift_ratio,
        "file_buffer_size": file_buffer_size,
        "file_parallelism": file_parallelism,
        "batch_size": batch_size,
        "compression_type": compression_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Cleanup created files
    for input_data in list_of_inputs:
        try:
            if "*" in input_data["file_pattern"]:
                files = glob.glob(input_data["file_pattern"])
                for file in files:
                    os.remove(file)
            else:
                os.remove(input_data["file_pattern"])
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"Error deleting file: {e}")


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RecordInput"] = tf_raw_ops_record_input_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RecordInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RecordInput'.")

check_valid('tf.raw_ops.RecordInput', generated_inputs['tf.raw_ops.RecordInput'], lib="tf", suffix=0)
