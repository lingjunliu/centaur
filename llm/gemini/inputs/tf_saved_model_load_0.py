
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

        # Create a dummy function and save it
        @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32, name='x')])
        def dummy_function(x):
            return x * 2.0

        concrete_function = dummy_function.get_concrete_function()
        tf.saved_model.save(
            obj=dummy_function,
            export_dir=export_dir,
            signatures={'serving_default': concrete_function}
        )

    # Input 1: Basic valid input with minimal parameters
    export_dir = "dummy_saved_model_1"
    create_dummy_saved_model(export_dir)
    tags = None
    options = None
    input_dict = {"export_dir": export_dir, "tags": tags, "options": str(options)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With a specific tag
    export_dir = "dummy_saved_model_2"
    create_dummy_saved_model(export_dir, tags=["serve"])
    tags = ["serve"]
    options = None
    input_dict = {"export_dir": export_dir, "tags": tags, "options": str(options)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty tag list
    export_dir = "dummy_saved_model_3"
    create_dummy_saved_model(export_dir)
    tags = []
    options = None
    input_dict = {"export_dir": export_dir, "tags": tags, "options": str(options)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple tags
    export_dir = "dummy_saved_model_4"
    create_dummy_saved_model(export_dir, tags=["serve", "train"])
    tags = ["serve", "train"]
    options = None
    input_dict = {"export_dir": export_dir, "tags": tags, "options": str(options)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Longer export directory path
    export_dir = "path/to/a/very/long/dummy_saved_model_5"
    if not os.path.exists(os.path.dirname(export_dir)):
        os.makedirs(os.path.dirname(export_dir))

    create_dummy_saved_model(export_dir)
    tags = None
    options = None
    input_dict = {"export_dir": export_dir, "tags": tags, "options": str(options)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Export dir with spaces
    export_dir = "dummy saved model 6"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    create_dummy_saved_model(export_dir)
    tags = None
    options = None
    input_dict = {"export_dir": export_dir, "tags": tags, "options": str(options)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Options specified
    export_dir = "dummy_saved_model_7"
    create_dummy_saved_model(export_dir)
    tags = None
    options = str(tf.saved_model.LoadOptions())
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different tags
    export_dir = "dummy_saved_model_8"
    create_dummy_saved_model(export_dir, tags=["serving"])
    tags = ["serving"]
    options = None
    input_dict = {"export_dir": export_dir, "tags": tags, "options": str(options)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: With a dot in directory name
    export_dir = "dummy.saved.model.9"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    create_dummy_saved_model(export_dir)
    tags = None
    options = None
    input_dict = {"export_dir": export_dir, "tags": tags, "options": str(options)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Options specified with different parameters
    export_dir = "dummy_saved_model_10"
    create_dummy_saved_model(export_dir)
    tags = None
    options = str(tf.saved_model.LoadOptions())
    input_dict = {"export_dir": export_dir, "tags": tags, "options": options}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.saved_model.load"] = tf_saved_model_load_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.saved_model.load' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.saved_model.load'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.saved_model.load', generated_inputs['tf.saved_model.load'], lib="tf", suffix=0)
