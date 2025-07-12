
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_csv_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    records = np.array(["1,2,3", "4,5,6"])
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different delimiter
    records = np.array(["1;2;3", "4;5;6"])
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ";"
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With quotes
    records = np.array(['"1","2","3"', '"4","5","6"'])
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: No quotes
    records = np.array(['"1","2","3"', '"4","5","6"'])
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = False
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With NA values
    records = np.array(["1,NA,3", "4,5,NA"])
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = "NA"
    select_cols = None
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: With selected columns
    records = np.array(["1,2,3", "4,5,6"])
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = [0, 2]
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different data types
    records = np.array(["1.1,a,2", "3.3,b,4"])
    record_defaults = [tf.constant(0.0, dtype=tf.float32), tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty records
    records = np.array(["", ""])
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: records with leading/trailing spaces
    records = np.array([" 1, 2 , 3 ", " 4 ,5, 6 "])
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String records with string defaults
    records = np.array(["a,b,c", "d,e,f"])
    record_defaults = [tf.constant("default", dtype=tf.string), tf.constant("default", dtype=tf.string), tf.constant("default", dtype=tf.string)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Different name
    records = np.array(["1,2,3", "4,5,6"])
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = "my_decode"
    input_dict = {"records": records, "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
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
