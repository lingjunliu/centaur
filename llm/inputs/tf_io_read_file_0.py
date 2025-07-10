
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import os
import tempfile

def tf_io_read_file_inputs():
    list_of_inputs = []

    # Create temporary files for testing
    file1 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt")
    with file1 as f1:
        f1.write("This is file 1.")
        file1_path = f1.name

    file2 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt")
    with file2 as f2:
        f2.write("Another file with more content.\nLine 2\nLine 3")
        file2_path = f2.name
        
    file3 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt")
    with file3 as f3:
        file3_path = f3.name # creates empty file by default

    file4 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt", encoding="utf-8")
    with file4 as f4:
        f4.write("你好世界")
        file4_path = f4.name

    file5 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt")
    with file5 as f5:
        long_string = "a" * 1024  # Reduced size to avoid potential issues
        f5.write(long_string)
        file5_path = f5.name

    file6 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt")
    with file6 as f6:
        f6.write("This is a file\nwith a newline.")
        file6_path = f6.name

    file7 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt")
    with file7 as f7:
        f7.write("~!@#$%^&*()_+=-`")
        file7_path = f7.name

    file8 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt")
    with file8 as f8:
        f8.write("1234567890")
        file8_path = f8.name
        
    file9 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt")
    with file9 as f9:
        f9.write("Mixed content 123 abc\n~!@#$")
        file9_path = f9.name
        
    file10 = tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".txt")
    with file10 as f10:
        f10.write("This\tis\ta\ttabbed\tfile")
        file10_path = f10.name


    # Input 1, valid
    input_dict = {
        "filename": file1_path,
        "name": "read_file_1"
    }
    list_of_inputs.append(input_dict)

    # Input 2, valid
    input_dict = {
        "filename": file2_path,
        "name": None
    }
    list_of_inputs.append(input_dict)
    
    # Input 3, valid: Empty file
    input_dict = {
        "filename": file3_path,
        "name": "read_empty_file"
    }
    list_of_inputs.append(input_dict)

    # Input 4, valid: Unicode characters
    input_dict = {
        "filename": file4_path,
        "name": "read_unicode_file"
    }
    list_of_inputs.append(input_dict)
    
    # Input 5, valid: Long file
    input_dict = {
        "filename": file5_path,
        "name": "read_long_file"
    }
    list_of_inputs.append(input_dict)

    # Input 6, valid: File with newline
    input_dict = {
        "filename": file6_path,
        "name": "read_newline_file"
    }
    list_of_inputs.append(input_dict)
    
    # Input 7, valid: File with special characters
    input_dict = {
        "filename": file7_path,
        "name": "read_special_chars_file"
    }
    list_of_inputs.append(input_dict)

    # Input 8, valid: File with numbers
    input_dict = {
        "filename": file8_path,
        "name": "read_numbers_file"
    }
    list_of_inputs.append(input_dict)
    
    # Input 9, valid: File with mixed content
    input_dict = {
        "filename": file9_path,
        "name": "read_mixed_content_file"
    }
    list_of_inputs.append(input_dict)

    # Input 10, valid: File with tabs
    input_dict = {
        "filename": file10_path,
        "name": "read_tabs_file"
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.read_file"] = tf_io_read_file_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.read_file' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.read_file'.")

check_valid('tf.io.read_file', generated_inputs['tf.io.read_file'], lib="tf", suffix=0)
