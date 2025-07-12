
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_io_gfile_rename_inputs():
    list_of_inputs = []

    # Create dummy files for testing
    os.makedirs("src_dir1", exist_ok=True)
    os.makedirs("src_path/to", exist_ok=True)
    os.makedirs("src_data", exist_ok=True)
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")

    with open("src_file1.txt", "w") as f:
        f.write("This is a test file.")
    with open("src_dir1/src_file1.txt", "w") as f:
        f.write("This is a test file.")
    with open("src_path/to/my_file.dat", "w") as f:
        f.write("This is a test file.")
    with open("src_old_file.log", "w") as f:
        f.write("This is a test file.")
    with open("src_data/src_input_file.json", "w") as f:
        f.write("This is a test file.")
    with open("src_file_to_move.pdf", "w") as f:
        f.write("This is a test file.")
    with open("src_very_long_file_name.txt", "w") as f:
        f.write("This is a test file.")
    os.makedirs("./src_data", exist_ok=True)
    with open("./src_data/src_file1.txt", "w") as f:
        f.write("This is a test file.")
    with open("src_source.csv", "w") as f:
        f.write("This is a test file.")

    with open("/tmp/src_temp_file.txt", "w") as f:
        f.write("This is a test file.")

    # Input 1
    src = "src_file1.txt"
    dst = "dst_file2.txt"
    overwrite = False
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    src = "src_dir1/src_file1.txt"
    dst = "dst_dir2/dst_file2.txt"
    overwrite = True
    os.makedirs("dst_dir2", exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    src = "src_path/to/my_file.dat"
    dst = "dst_new_location/my_file.dat"
    overwrite = False
    os.makedirs("dst_new_location", exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    src = "src_old_file.log"
    dst = "dst_archive/old_file.log"
    overwrite = True
    os.makedirs("dst_archive", exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    src = "/tmp/src_temp_file.txt"
    dst = "/tmp/dst_final_file.txt"
    overwrite = False
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    src = "src_source.csv"
    dst = "dst_destination.csv"
    overwrite = True
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    src = "src_data/src_input_file.json"
    dst = "dst_output/dst_processed_file.json"
    overwrite = False
    os.makedirs("dst_output", exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    src = "src_file_to_move.pdf"
    dst = "dst_new_folder/file_to_move.pdf"
    overwrite = True
    os.makedirs("dst_new_folder", exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    src = "src_very_long_file_name.txt"
    dst = "dst_short_name.txt"
    overwrite = False
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    src = "./src_data/src_file1.txt"
    dst = "../dst_backup/file1.txt"
    overwrite = True
    os.makedirs("../dst_backup", exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.rename"] = tf_io_gfile_rename_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.rename' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.rename'.")

check_valid('tf.io.gfile.rename', generated_inputs['tf.io.gfile.rename'], lib="tf", suffix=0)
