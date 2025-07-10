
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os
import shutil

def tf_io_gfile_rename_inputs():
    list_of_inputs = []

    # Create dummy files and directories for testing
    if not os.path.exists("source_file_1.txt"):
        open("source_file_1.txt", "w").close()
    if os.path.exists("dest_file_1.txt"):
        os.remove("dest_file_1.txt")
    if not os.path.exists("dest_file_2.txt"):
        open("dest_file_2.txt", "w").close()
    if not os.path.exists("source_file_2.txt"):
        open("source_file_2.txt", "w").close()
    if not os.path.exists("source_dir_1"):
        os.makedirs("source_dir_1", exist_ok=True)
    if os.path.exists("dest_dir_1"):
        shutil.rmtree("dest_dir_1")
    if not os.path.exists("dest_dir_2"):
        os.makedirs("dest_dir_2", exist_ok=True)
    if not os.path.exists("source_dir_2"):
        os.makedirs("source_dir_2", exist_ok=True)
    if not os.path.exists("parent_dir"):
        os.makedirs("parent_dir", exist_ok=True)
    if not os.path.exists("similar_name_src.txt"):
        open("similar_name_src.txt", "w").close()
    if not os.path.exists("same_name.txt"):
        open("same_name.txt", "w").close()
    if not os.path.exists("source.txt"):
        open("source.txt", "w").close()
    if not os.path.exists("very_long_source_file_name_with_many_characters.txt"):
      open("very_long_source_file_name_with_many_characters.txt", "w").close()
    if os.path.exists("parent_dir/child_dir"):
        shutil.rmtree("parent_dir/child_dir")




    # Input 1: Basic rename
    src = np.str_("source_file_1.txt")
    dst = np.str_("dest_file_1.txt")
    overwrite = np.bool_(False)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Overwrite existing destination
    src = np.str_("source_file_2.txt")
    dst = np.str_("dest_file_2.txt")
    overwrite = np.bool_(True)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rename directory
    src = np.str_("source_dir_1")
    dst = np.str_("dest_dir_1")
    overwrite = np.bool_(False)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Overwrite existing directory
    src = np.str_("source_dir_2")
    dst = np.str_("dest_dir_2")
    overwrite = np.bool_(True)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  Rename with similar names
    src = np.str_("similar_name_src.txt")
    dst = np.str_("similar_name_dst.txt")
    overwrite = np.bool_(False)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6:  Destination is a subdirectory of source
    src = np.str_("parent_dir")
    dst = np.str_("parent_dir/child_dir")
    overwrite = np.bool_(True)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rename with a long file name
    src = np.str_("very_long_source_file_name_with_many_characters.txt")
    dst = np.str_("very_long_dest_file_name_with_many_characters.txt")
    overwrite = np.bool_(False)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Empty source name
    src = np.str_("")
    dst = np.str_("destination.txt")
    overwrite = np.bool_(False)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Empty destination name
    src = np.str_("source.txt")
    dst = np.str_("")
    overwrite = np.bool_(True)
    input_dict = {"src": src, "dst": dst, "overwrite": overwrite}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10:  Rename to the same name
    src = np.str_("same_name.txt")
    dst = np.str_("same_name.txt")
    overwrite = np.bool_(False)
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
