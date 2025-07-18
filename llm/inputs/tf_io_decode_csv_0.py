
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_io_decode_csv_inputs():
    list_of_inputs = []

    # Input 1: Basic case, all required fields
    input_dict_1 = {
        'records': np.array(['1,2.0,hello', '4,5.1,world'], dtype=object),
        'record_defaults': [np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.string_)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: With default values for missing fields
    input_dict_2 = {
        'records': np.array(['1,,hello', '4,5.1,'], dtype=object),
        'record_defaults': [np.array(0, dtype=np.int32), np.array(0.0, dtype=np.float32), np.array('NA', dtype=np.string_)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'with_defaults'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Different delimiter
    input_dict_3 = {
        'records': np.array(['1|2.0|hello', '4|5.1|world'], dtype=object),
        'record_defaults': [np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.string_)],
        'field_delim': '|',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'different_delim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Handling quoted fields containing delimiters
    input_dict_4 = {
        'records': np.array(['1,"a,b",3.0', '2,"c,d",4.0'], dtype=object),
        'record_defaults': [np.array([], dtype=np.int32), np.array([], dtype=np.string_), np.array([], dtype=np.float32)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'quoted_fields'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Custom NA value
    input_dict_5 = {
        'records': np.array(['1,N/A,hello', '4,5.1,world'], dtype=object),
        'record_defaults': [np.array(0, dtype=np.int32), np.array(0.0, dtype=np.float32), np.array('default', dtype=np.string_)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'N/A',
        'select_cols': None,
        'name': 'custom_na'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty records tensor
    input_dict_6 = {
        'records': np.array([], dtype=object),
        'record_defaults': [np.array([], dtype=np.int64), np.array([], dtype=np.float64)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'empty_records'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: use_quote_delim=False
    input_dict_7 = {
        'records': np.array(['1,"a,b",3.0'], dtype=object),
        'record_defaults': [np.array(0, dtype=np.int32), np.array('', dtype=np.string_), np.array('', dtype=np.string_), np.array('', dtype=np.string_), np.array(0.0, dtype=np.float32)],
        'field_delim': ',',
        'use_quote_delim': False,
        'na_value': '',
        'select_cols': None,
        'name': 'no_quote_delim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Leading/trailing spaces in numeric fields
    input_dict_8 = {
        'records': np.array([' 1  ,  2.5  ', '-10 , -3.14'], dtype=object),
        'record_defaults': [np.array([], dtype=np.int32), np.array([], dtype=np.float64)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'with_spaces'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using select_cols
    input_dict_9 = {
        'records': np.array(['a,1,1.0,x', 'b,2,2.0,y'], dtype=object),
        'record_defaults': [np.array('', dtype=np.string_), np.array(0.0, dtype=np.float32)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': [0, 2],
        'name': 'select_cols'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Single column
    input_dict_10 = {
        'records': np.array(['-1.1', '2.2', '3.3'], dtype=object),
        'record_defaults': [np.array(0.0, dtype=np.float32)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'single_column'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

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

check_valid('tf.io.decode_csv', generated_inputs['tf.io.decode_csv'], lib="tf", suffix=0)
