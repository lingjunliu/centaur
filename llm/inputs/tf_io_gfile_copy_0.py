
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_io_gfile_copy_inputs():
    list_of_inputs = []

    # Input 1: Basic copy
    src = "/tmp/src_file1.txt"
    dst = "/tmp/dst_file1.txt"
    overwrite = False
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists(dst):
            os.remove(dst)
        with open(src, "w") as f:
            f.write("Test data 1")
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except Exception as e:
        print(f"Error creating or deleting file: {e}")
        

    # Input 2: Overwrite existing file
    src = "/tmp/src_file2.txt"
    dst = "/tmp/dst_file2.txt"
    overwrite = True
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists(dst):
            os.remove(dst)
        with open(src, "w") as f:
            f.write("Test data 2")
        with open(dst, "w") as f:
            f.write("Initial data")
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except Exception as e:
        print(f"Error creating or deleting file: {e}")

    # Input 3: Copy with different URI scheme
    src = "/tmp/src_file3.txt"
    dst = "file:///tmp/dst_file3.txt"
    overwrite = False
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists("/tmp/dst_file3.txt"):
            os.remove("/tmp/dst_file3.txt")
        with open(src, "w") as f:
            f.write("Test data 3")
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except Exception as e:
        print(f"Error creating or deleting file: {e}")

    # Input 4: Copy to a subdirectory (requires file name in dst)
    src = "/tmp/src_file4.txt"
    dst = "/tmp/new_dir/dst_file4.txt"
    overwrite = False
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists("/tmp/new_dir/dst_file4.txt"):
            os.remove("/tmp/new_dir/dst_file4.txt")

        os.makedirs("/tmp/new_dir", exist_ok=True)
        with open(src, "w") as f:
            f.write("Test data 4")
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))

    except Exception as e:
        print(f"Error creating or deleting file: {e}")

    # Input 5: Long file paths
    src = "/tmp/" + "a" * 200 + "src_file5.txt"
    dst = "/tmp/" + "b" * 200 + "dst_file5.txt"
    overwrite = True
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists(dst):
            os.remove(dst)
        with open(src, "w") as f:
            f.write("Test data 5")
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))

    except Exception as e:
        print(f"Error creating or deleting file: {e}")

    # Input 6: Empty file
    src = "/tmp/src_file6.txt"
    dst = "/tmp/dst_file6.txt"
    overwrite = False
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists(dst):
            os.remove(dst)
        open(src, "w").close()  # Create an empty file
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))

    except Exception as e:
        print(f"Error creating or deleting file: {e}")

   # Input 7:  File with special characters
    src = "/tmp/src_file7.txt"
    dst = "/tmp/dst_file7.txt"
    overwrite = False
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists(dst):
            os.remove(dst)
        with open(src, "w") as f:
            f.write("!@#$%^&*()_+=-`~[]{}|;':\",./<>?")
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))

    except Exception as e:
        print(f"Error creating or deleting file: {e}")

    # Input 8:  Another path
    src = "/tmp/src_file8.txt"
    dst = "/tmp/dst_file8.txt"
    overwrite = True
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists(dst):
            os.remove(dst)
        with open(src, "w") as f:
            f.write("Test data 8")
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except Exception as e:
        print(f"Error creating or deleting file: {e}")

    # Input 9: Different dst scheme
    src = "/tmp/src_file9.txt"
    dst = "file:///tmp/dst_file9.txt"
    overwrite = False
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists("/tmp/dst_file9.txt"):
            os.remove("/tmp/dst_file9.txt")
        with open(src, "w") as f:
            f.write("Test data 9")
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except Exception as e:
        print(f"Error creating or deleting file: {e}")

    # Input 10: Overwrite True on a non-existent dst
    src = "/tmp/src_file10.txt"
    dst = "/tmp/dst_file10.txt"
    overwrite = True
    try:
        if os.path.exists(src):
            os.remove(src)
        if os.path.exists(dst):
            os.remove(dst)
        with open(src, "w") as f:
            f.write("Test data 10")
        input_dict = {"src": src, "dst": dst, "overwrite": np.bool_(overwrite)}
        list_of_inputs.append(copy.deepcopy(input_dict))
    except Exception as e:
        print(f"Error creating or deleting file: {e}")

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.copy"] = tf_io_gfile_copy_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.copy'.")

check_valid('tf.io.gfile.copy', generated_inputs['tf.io.gfile.copy'], lib="tf", suffix=0)
