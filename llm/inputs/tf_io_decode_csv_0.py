
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_csv_inputs():
    list_of_inputs = []

    # Input 1
    records = tf.constant(["1,2,3", "4,5,6"])
    record_defaults = [tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    records = tf.constant(["1.1,2.2,3.3", "4.4,5.5,6.6"])
    record_defaults = [tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    records = tf.constant(["a,b,c", "d,e,f"])
    record_defaults = [tf.constant([""], dtype=tf.string), tf.constant([""], dtype=tf.string), tf.constant([""], dtype=tf.string)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    records = tf.constant(["1,2", "3,4"])
    record_defaults = [tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = [0]
    name = None

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    records = tf.constant(["1;2;3", "4;5;6"])
    record_defaults = [tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32)]
    field_delim = ";"
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    records = tf.constant(["\"1\",\"2\",\"3\"", "\"4\",\"5\",\"6\""])
    record_defaults = [tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    records = tf.constant(["1,2,NA", "4,NA,6"])
    record_defaults = [tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = "NA"
    select_cols = None
    name = None

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    records = tf.constant(["\"hello, world\",\"123\"", "\"goodbye, world\",\"456\""])
    record_defaults = [tf.constant([""], dtype=tf.string), tf.constant([""], dtype=tf.string)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    records = tf.constant(["1,2,3", "4,5,6"])
    record_defaults = [tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32), tf.constant([0], dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = False
    na_value = ""
    select_cols = None
    name = "my_decode_csv"

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    records = tf.constant(["1,2,3,4,5", "6,7,8,9,10"])
    record_defaults = [tf.constant([0], dtype=tf.int32)] * 5
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = [0, 2, 4]
    name = None

    input_dict = {
        "records": records,
        "record_defaults": record_defaults,
        "field_delim": field_delim,
        "use_quote_delim": use_quote_delim,
        "na_value": na_value,
        "select_cols": select_cols,
        "name": name
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
