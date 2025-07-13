
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_saved_model_load_inputs():
    list_of_inputs = []

    # Helper function to create a dummy SavedModel directory
    def create_dummy_saved_model(export_dir, tags=None):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        signatures = {'serving_default': tf.function(lambda x: x, input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])}
        tf.saved_model.save(tf.train.Checkpoint(), export_dir, signatures=signatures)

    # Input 1: Basic case with empty options
    export_dir = "dummy_saved_model_1"
    create_dummy_saved_model(export_dir)
    input_dict = {"export_dir": export_dir, "tags": [], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With tags
    export_dir = "dummy_saved_model_2"
    create_dummy_saved_model(export_dir, tags=["serve", "train"])
    input_dict = {"export_dir": export_dir, "tags": ["serve"], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With options (empty string as LoadOptions object can't be created without TF)
    export_dir = "dummy_saved_model_3"
    create_dummy_saved_model(export_dir)
    input_dict = {"export_dir": export_dir, "tags": [], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different tags
    export_dir = "dummy_saved_model_4"
    create_dummy_saved_model(export_dir, tags=["tag1", "tag2"])
    input_dict = {"export_dir": export_dir, "tags": ["tag2"], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple tags
    export_dir = "dummy_saved_model_5"
    create_dummy_saved_model(export_dir, tags=["tag1", "tag2", "tag3"])
    input_dict = {"export_dir": export_dir, "tags": ["tag1", "tag3"], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: No tags specified when multiple exist
    export_dir = "dummy_saved_model_6"
    create_dummy_saved_model(export_dir, tags=["tag1", "tag2"])
    input_dict = {"export_dir": export_dir, "tags": [], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Export Dir with Sub directory.
    export_dir = "dummy_saved_model_7/sub_dir"
    create_dummy_saved_model(export_dir)
    input_dict = {"export_dir": export_dir, "tags": [], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty Tags list
    export_dir = "dummy_saved_model_8"
    create_dummy_saved_model(export_dir)
    input_dict = {"export_dir": export_dir, "tags": [], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another set of tags
    export_dir = "dummy_saved_model_9"
    create_dummy_saved_model(export_dir, tags=["gpu", "cpu"])
    input_dict = {"export_dir": export_dir, "tags": ["cpu"], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different Export Directory Name.
    export_dir = "my_saved_model"
    create_dummy_saved_model(export_dir)
    input_dict = {"export_dir": export_dir, "tags": [], "options": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for i in range(1, 11):
        export_dir = f"dummy_saved_model_{i}"
        if os.path.exists(export_dir):
            import shutil
            shutil.rmtree(export_dir, ignore_errors=True)

    if os.path.exists("my_saved_model"):
        import shutil
        shutil.rmtree("my_saved_model", ignore_errors=True)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.saved_model.load"] = tf_saved_model_load_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.saved_model.load' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.saved_model.load'.")

check_valid('tf.saved_model.load', generated_inputs['tf.saved_model.load'], lib="tf", suffix=0)
