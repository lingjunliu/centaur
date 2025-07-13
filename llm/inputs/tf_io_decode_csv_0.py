
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_csv_inputs():
    list_of_inputs = []

    # Input 1: Basic case with different data types
    records = tf.constant(["1.0,2,hello", "3.0,4,world"])
    record_defaults = [tf.constant(1.0), tf.constant(2), tf.constant("hello")]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ",",
        "use_quote_delim": True,
        "na_value": "",
        "select_cols": None,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different field delimiter and na_value
    records = tf.constant(["1.0;2;hello", "3.0;4;NA"])
    record_defaults = [tf.constant(1.0), tf.constant(2), tf.constant("hello")]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ";",
        "use_quote_delim": True,
        "na_value": "NA",
        "select_cols": None,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: No quote delimiter
    records = tf.constant(["\"1.0\",\"2\",\"hello\"", "\"3.0\",\"4\",\"world\""])
    record_defaults = [tf.constant(1.0), tf.constant(2), tf.constant("hello")]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ",",
        "use_quote_delim": False,
        "na_value": "",
        "select_cols": None,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Select specific columns
    records = tf.constant(["1.0,2,hello,4.0", "3.0,4,world,6.0"])
    record_defaults = [tf.constant(1.0), tf.constant(2), tf.constant("hello"), tf.constant(4.0)]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ",",
        "use_quote_delim": True,
        "na_value": "",
        "select_cols": [0, 2],
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Empty records
    records = tf.constant(["", ""])
    record_defaults = [tf.constant(1.0), tf.constant(2)]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ",",
        "use_quote_delim": True,
        "na_value": "",
        "select_cols": None,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple default values - Removing it as there's only one type of tensor accepted according to signature
    records = tf.constant(["1.0,2,hello", "3.0,4,world"])
    record_defaults = [tf.constant(1.0), tf.constant(2), tf.constant("hello")]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ",",
        "use_quote_delim": True,
        "na_value": "",
        "select_cols": None,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values
    records = tf.constant(["-1.0,-2,hello", "-3.0,-4,world"])
    record_defaults = [tf.constant(-1.0), tf.constant(-2), tf.constant("hello")]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ",",
        "use_quote_delim": True,
        "na_value": "",
        "select_cols": None,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Long strings, different lengths
    records = tf.constant(["verylongstring1,2,hello", "short,4,verylongstring2"])
    record_defaults = [tf.constant("verylongstring1"), tf.constant(2), tf.constant("hello")]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ",",
        "use_quote_delim": True,
        "na_value": "",
        "select_cols": None,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float64 and Int64 - Removing it as there's only one type of tensor accepted according to signature
    records = tf.constant(["1.0,2", "3.0,4"])
    record_defaults = [tf.constant(1.0), tf.constant(2)]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ",",
        "use_quote_delim": True,
        "na_value": "",
        "select_cols": None,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: With name
    records = tf.constant(["1.0,2,hello", "3.0,4,world"])
    record_defaults = [tf.constant(1.0), tf.constant(2), tf.constant("hello")]
    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": ",",
        "use_quote_delim": True,
        "na_value": "",
        "select_cols": None,
        "name": "my_decode_csv"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
