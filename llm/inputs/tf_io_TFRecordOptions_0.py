
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_tfrecordoptions_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'compression_type': np.string_(""),
        'flush_mode': np.string_(""),
        'input_buffer_size': np.int32(0),
        'output_buffer_size': np.int32(0),
        'window_bits': np.int32(0),
        'compression_level': np.int32(0),
        'compression_method': np.int32(0),
        'mem_level': np.int32(0),
        'compression_strategy': np.int32(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'compression_type': np.string_("ZLIB"),
        'flush_mode': np.string_(""),
        'input_buffer_size': np.int32(1024),
        'output_buffer_size': np.int32(1024),
        'window_bits': np.int32(15),
        'compression_level': np.int32(-1),
        'compression_method': np.int32(8),
        'mem_level': np.int32(8),
        'compression_strategy': np.int32(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'compression_type': np.string_("GZIP"),
        'flush_mode': np.string_(""),
        'input_buffer_size': np.int32(4096),
        'output_buffer_size': np.int32(4096),
        'window_bits': np.int32(9),
        'compression_level': np.int32(5),
        'compression_method': np.int32(8),
        'mem_level': np.int32(9),
        'compression_strategy': np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'compression_type': np.string_(""),
        'flush_mode': np.string_("FLUSH_BLOCK"),
        'input_buffer_size': np.int32(8192),
        'output_buffer_size': np.int32(8192),
        'window_bits': np.int32(12),
        'compression_level': np.int32(9),
        'compression_method': np.int32(8),
        'mem_level': np.int32(5),
        'compression_strategy': np.int32(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'compression_type': np.string_("ZLIB"),
        'flush_mode': np.string_("SYNC_FLUSH"),
        'input_buffer_size': np.int32(16384),
        'output_buffer_size': np.int32(16384),
        'window_bits': np.int32(10),
        'compression_level': np.int32(1),
        'compression_method': np.int32(8),
        'mem_level': np.int32(2),
        'compression_strategy': np.int32(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'compression_type': np.string_("GZIP"),
        'flush_mode': np.string_(""),
        'input_buffer_size': np.int32(32768),
        'output_buffer_size': np.int32(32768),
        'window_bits': np.int32(11),
        'compression_level': np.int32(6),
        'compression_method': np.int32(8),
        'mem_level': np.int32(1),
        'compression_strategy': np.int32(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'compression_type': np.string_(""),
        'flush_mode': np.string_(""),
        'input_buffer_size': np.int32(65536),
        'output_buffer_size': np.int32(65536),
        'window_bits': np.int32(13),
        'compression_level': np.int32(2),
        'compression_method': np.int32(8),
        'mem_level': np.int32(3),
        'compression_strategy': np.int32(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'compression_type': np.string_("ZLIB"),
        'flush_mode': np.string_("FLUSH_BLOCK"),
        'input_buffer_size': np.int32(131072),
        'output_buffer_size': np.int32(131072),
        'window_bits': np.int32(14),
        'compression_level': np.int32(3),
        'compression_method': np.int32(8),
        'mem_level': np.int32(4),
        'compression_strategy': np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_dict = {
        'compression_type': np.string_("GZIP"),
        'flush_mode': np.string_("SYNC_FLUSH"),
        'input_buffer_size': np.int32(262144),
        'output_buffer_size': np.int32(262144),
        'window_bits': np.int32(15),
        'compression_level': np.int32(4),
        'compression_method': np.int32(8),
        'mem_level': np.int32(6),
        'compression_strategy': np.int32(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'compression_type': np.string_(""),
        'flush_mode': np.string_(""),
        'input_buffer_size': np.int32(524288),
        'output_buffer_size': np.int32(524288),
        'window_bits': np.int32(8),
        'compression_level': np.int32(7),
        'compression_method': np.int32(8),
        'mem_level': np.int32(7),
        'compression_strategy': np.int32(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.TFRecordOptions"] = tf_io_tfrecordoptions_inputs()
for i in range(len(generated_inputs["tf.io.TFRecordOptions"])):
    for key in generated_inputs["tf.io.TFRecordOptions"][i].keys():
        if isinstance(generated_inputs["tf.io.TFRecordOptions"][i][key], np.string_):
            generated_inputs["tf.io.TFRecordOptions"][i][key] = generated_inputs["tf.io.TFRecordOptions"][i][key].decode('utf-8')

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.TFRecordOptions' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.TFRecordOptions'.")

check_valid('tf.io.TFRecordOptions', generated_inputs['tf.io.TFRecordOptions'], lib="tf", suffix=0)
