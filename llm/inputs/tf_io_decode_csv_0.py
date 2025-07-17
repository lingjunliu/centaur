
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_csv_inputs():
    list_of_inputs = []

    def create_input_dict(records, record_defaults, field_delim=",", use_quote_delim=True, na_value="", select_cols=None, name=None):
        return {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}

    # Input 1: Basic CSV parsing
    records = tf.constant("1,2,3\n4,5,6")
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults)))

    # Input 2: Different delimiter
    records = tf.constant("1;2;3\n4;5;6")
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults, field_delim=";")))

    # Input 3: Float type
    records = tf.constant("1.1,2.2,3.3\n4.4,5.5,6.6")
    record_defaults = [tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults)))

    # Input 4: NA value
    records = tf.constant("1,NA,2\n3,4,NA")
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults, na_value="NA")))

    # Input 5: Select columns
    records = tf.constant("1,2,3\n4,5,6")
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults, select_cols=[0, 2])))

    # Input 6: No quote delimiter
    records = tf.constant('"1","2","3"\n"4","5","6"')
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults, use_quote_delim=False)))

    # Input 7: Empty values with defaults
    records = tf.constant(",,\n,1,")
    record_defaults = [tf.constant(1, dtype=tf.int32), tf.constant(2, dtype=tf.int32), tf.constant(3, dtype=tf.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults)))

    # Input 8: Int64 type
    records = tf.constant("10000000000,20000000000,30000000000\n40000000000,50000000000,60000000000")
    record_defaults = [tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults)))

    # Input 9: Float64 type
    records = tf.constant("1.1,2.2,3.3\n4.4,5.5,6.6")
    record_defaults = [tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults)))

    # Input 10: Empty records
    records = tf.constant("")
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(records, record_defaults)))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_csv"] = tf_io_decode_csv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_csv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_csv'.")

check_valid('tf.io.decode_csv', generated_inputs['tf.io.decode_csv'], lib="tf", suffix=0)
