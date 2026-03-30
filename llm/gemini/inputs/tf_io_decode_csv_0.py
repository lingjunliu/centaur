
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_csv_inputs():
    list_of_inputs = []

    # Input 1
    records = np.array(["1,2,3", "4,5,6", "7,8,9"], dtype=np.string_)
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    records = np.array(["1.1,2.2,3.3", "4.4,5.5,6.6"], dtype=np.string_)
    record_defaults = [tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    records = np.array(["a,b,c", "d,e,f"], dtype=np.string_)
    record_defaults = [tf.constant("default", dtype=tf.string), tf.constant("default", dtype=tf.string), tf.constant("default", dtype=tf.string)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    records = np.array(["1,2,3", "4,5,6"], dtype=np.string_)
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = [0, 1]
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    records = np.array(["1;2;3", "4;5;6"], dtype=np.string_)
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ";"
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    records = np.array(["\"1,2\",3", "\"4,5\",6"], dtype=np.string_)
    record_defaults = [tf.constant("default", dtype=tf.string), tf.constant("default", dtype=tf.string)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    records = np.array(["1,NA,3", "4,5,6"], dtype=np.string_)
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = "NA"
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    records = np.array(["1,2,3", "4,5,6"], dtype=np.string_)
    record_defaults = [tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    records = np.array(["1,2", "4,5,6,7"], dtype=np.string_)
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    records = np.array(["\"hello world\",2,3", "4,5,6"], dtype=np.string_)
    record_defaults = [tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    records = np.array(["1,2,3", "4,5,6"], dtype=np.string_)
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = False
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    records = np.array(["1,2", "4,5"], dtype=np.string_)
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(9, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    records = np.array(["\"hello, world\",2,3", "4,5,6"], dtype=np.string_)
    record_defaults = [tf.constant("default", dtype=tf.string), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14
    records = np.array([b"1,2,3", b"4,5,6"], dtype=np.string_)
    record_defaults = [tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)]
    field_delim = ","
    use_quote_delim = True
    na_value = ""
    select_cols = None
    name = None
    input_dict = {"records": records.astype(np.string_), "record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_csv"] = tf_io_decode_csv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.decode_csv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_csv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.io.decode_csv', generated_inputs['tf.io.decode_csv'], lib="tf", suffix=0)
