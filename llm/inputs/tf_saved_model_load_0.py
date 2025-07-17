
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os
import shutil

def tf_saved_model_load_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input.  Assume 'test_model_1' exists
    export_dir = "test_model_1"
    tags = None
    options = None
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        tf.saved_model.save(model, export_dir, signatures={})


    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)


    # Input 2: With tags, assuming 'test_model_2' exists
    export_dir = "test_model_2"
    tags = ["serve"]
    options = None

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)])
        def serve(x):
            return model(x)
        tf.saved_model.save(model, export_dir, signatures={"serving_default": serve})
    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    # Input 3: With empty tags list, assuming 'test_model_3' exists
    export_dir = "test_model_3"
    tags = []
    options = None
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        tf.saved_model.save(model, export_dir, signatures={})

    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    # Input 4: Using LoadOptions (string), assuming 'test_model_4' exists.  Note: LoadOptions must be specified
    # as a string due to the signature.

    export_dir = "test_model_4"
    tags = None
    options = ""  # Represents default options since signature specifies string type
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        tf.saved_model.save(model, export_dir, signatures={})


    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    # Input 5:  Multiple tags, assuming 'test_model_5' exists

    export_dir = "test_model_5"
    tags = ["serve", "train"]
    options = None
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        tf.saved_model.save(model, export_dir, tags=set(tags), signatures={})

    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)


    # Input 6: Another set of tags, assuming 'test_model_6' exists
    export_dir = "test_model_6"
    tags = ["gpu"]
    options = None
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        tf.saved_model.save(model, export_dir, signatures={})

    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)


    # Input 7: Multiple tags, assuming 'test_model_7' exists, empty string for options
    export_dir = "test_model_7"
    tags = ["serve", "validation"]
    options = ""
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        tf.saved_model.save(model, export_dir, signatures={})

    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)


    # Input 8:  Another tag, assuming 'test_model_8' exists, default options
    export_dir = "test_model_8"
    tags = ["inference"]
    options = None
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        tf.saved_model.save(model, export_dir, signatures={})

    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    # Input 9: Empty tag list with non-empty options string
    export_dir = "test_model_9"
    tags = []
    options = ""
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        tf.saved_model.save(model, export_dir, signatures={})

    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

     # Input 10: None tag list with non-empty options string
    export_dir = "test_model_10"
    tags = None
    options = ""
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(10, activation='relu')])
        tf.saved_model.save(model, export_dir, signatures={})

    input_dict = {
        "export_dir": export_dir,
        "tags": tags,
        "options": options
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

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
