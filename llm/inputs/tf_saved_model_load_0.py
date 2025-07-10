
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_saved_model_load_inputs():
    list_of_inputs = []

    # Create a dummy SavedModel for testing.
    def create_dummy_saved_model(export_dir):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        # Define a simple function
        def simple_function(x):
            return tf.reduce_sum(x, axis=1)

        # Convert it to a tf.function
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)])
        def tf_simple_function(x):
            return simple_function(x)
            
        # Save the tf.function
        tf.saved_model.save(
            obj=tf_simple_function,
            export_dir=export_dir,
            signatures={'serving_default': tf_simple_function.get_concrete_function()}
        )

    # Input 1
    export_dir = "./saved_model_1"
    create_dummy_saved_model(export_dir)
    tags = []
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    export_dir = "./saved_model_2"
    create_dummy_saved_model(export_dir)
    tags = ["serve"]
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    export_dir = "./saved_model_3"
    create_dummy_saved_model(export_dir)
    tags = ["train", "eval"]
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    export_dir = "./saved_model_4"
    create_dummy_saved_model(export_dir)
    tags = ["serving_default"]
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    export_dir = "./saved_model_5"
    create_dummy_saved_model(export_dir)
    tags = ["tag1", "tag2", "tag3"]
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    export_dir = "./saved_model_6"
    create_dummy_saved_model(export_dir)
    tags = [""] # An empty string tag is also accepted
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    export_dir = "./saved_model_7"
    create_dummy_saved_model(export_dir)
    tags = ["custom_tag"]
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    export_dir = "./saved_model_8"
    create_dummy_saved_model(export_dir)
    tags = ["", "another_tag"]
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    export_dir = "./saved_model_9"
    create_dummy_saved_model(export_dir)
    tags = ["special_tag", ""]
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    export_dir = "./saved_model_10"
    create_dummy_saved_model(export_dir)
    tags = ["", "", ""]
    options = ""
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
