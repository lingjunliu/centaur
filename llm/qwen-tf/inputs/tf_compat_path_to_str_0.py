
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_compat_path_to_str_inputs():
    list_of_inputs = []

    path = r"C:\XYZ\tensorflow\./.././tensorflow"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = r"\\Server\Share\Folder\file.txt"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = r"D:\path with spaces\sub dir\file name.txt"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "./.././Corpus"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "/var/log/../tmp//./app/"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "."
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = ".."
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "~/.cache/pip"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "relative/path/with//double///slashes"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "archive.tar.gz"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = ""
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = ".env"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "C:/Windows/System32/drivers/etc/hosts"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "/home/用户/项目/数据集"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "../../..//folder/./subfolder/../file"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    return list_of_inputs

generated_inputs["tf.compat.path_to_str"] = tf_compat_path_to_str_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.compat.path_to_str' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.path_to_str'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.compat.path_to_str', generated_inputs['tf.compat.path_to_str'], lib="tf", suffix=0)
