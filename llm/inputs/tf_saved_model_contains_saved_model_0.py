
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os
import shutil

class DummyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.v = tf.Variable(1.0)

def tf_saved_model_contains_saved_model_inputs():
    list_of_inputs = []

    # Input 1: Empty directory
    export_dir = "empty_dir"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.rmdir(export_dir)

    # Input 2: Directory with only a non-SavedModel file
    export_dir = "non_saved_model_dir"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    with open(os.path.join(export_dir, "dummy.txt"), "w") as f:
        f.write("This is a dummy file.")
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.remove(os.path.join(export_dir, "dummy.txt"))
    os.rmdir(export_dir)

    # Input 3: Directory with a SavedModel (minimal)
    export_dir = "saved_model_dir_minimal"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    module = DummyModule()
    tf.saved_model.save(module, export_dir)
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shutil.rmtree(export_dir)

    # Input 4: Directory with a SavedModel and other files
    export_dir = "saved_model_dir_mixed"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    module = DummyModule()
    tf.saved_model.save(module, export_dir)
    with open(os.path.join(export_dir, "extra.txt"), "w") as f:
        f.write("Extra file.")
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shutil.rmtree(export_dir)

    # Input 5: Non-existent directory
    export_dir = "non_existent_dir"
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Relative path
    export_dir = "./relative_path"
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path containing spaces
    export_dir = "path with spaces"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    module = DummyModule()
    tf.saved_model.save(module, export_dir)
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shutil.rmtree(export_dir)

    # Input 8: Long path name
    export_dir = "very_long_directory_name_" * 10
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    module = DummyModule()
    tf.saved_model.save(module, export_dir)
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shutil.rmtree(export_dir)

    # Input 9: Path with special characters
    export_dir = "special_chars!@#$%^&*()"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    module = DummyModule()
    tf.saved_model.save(module, export_dir)
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shutil.rmtree(export_dir)

    # Input 10: Nested directory
    export_dir = os.path.join("nested", "directory")
    if not os.path.exists(os.path.dirname(export_dir)):
        os.makedirs(os.path.dirname(export_dir))
    module = DummyModule()
    tf.saved_model.save(module, export_dir)
    input_dict = {"export_dir": export_dir}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shutil.rmtree(os.path.dirname(export_dir))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.saved_model.contains_saved_model"] = tf_saved_model_contains_saved_model_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.saved_model.contains_saved_model' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.saved_model.contains_saved_model'.")

check_valid('tf.saved_model.contains_saved_model', generated_inputs['tf.saved_model.contains_saved_model'], lib="tf", suffix=0)
