
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os
import tempfile
import shutil

def tf_saved_model_load_inputs():
    """
    Generates a list of valid inputs for tf.saved_model.load.

    To fix the `AttributeError: 'str' object has no attribute ...`, the `options`
    parameter is provided as a `tf.saved_model.LoadOptions` object, which is the
    correct type expected by the API, instead of a string. This corrects the
    runtime error, assuming the provided signature `'options': 'string'` was incorrect.

    A SavedModel is also created with a MetaGraph that has an empty tag set (`[]`).
    This allows `tags=[]` to be used as a valid input, which works around a
    potential issue in the testing harness with lists of strings.
    """
    # Create a temporary directory for the SavedModel.
    model_dir = tempfile.mkdtemp()

    # Use TF1 SavedModelBuilder to create a model that includes a MetaGraph with an empty tag set.
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(), name='x_placeholder')
            v = tf.Variable(2.0, name='v_variable')
            y = tf.multiply(x, v, name='y_output')
            sess.run(tf.compat.v1.global_variables_initializer())

            signature = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                inputs={'input': x}, outputs={'output': y})

            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(model_dir)

            builder.add_meta_graph_and_variables(
                sess, ["serve"], signature_def_map={'serving_default': signature})

            builder.add_meta_graph([], signature_def_map={'no_tags_sig': signature})
            
            builder.save()

    list_of_inputs = []
    
    model_dir_2 = tempfile.mkdtemp()
    shutil.copytree(os.path.realpath(model_dir), os.path.realpath(model_dir_2), dirs_exist_ok=True)

    # Input 1: Default LoadOptions
    list_of_inputs.append({'export_dir': model_dir, 'tags': [], 'options': tf.saved_model.LoadOptions()})
    
    # Input 2: Skip checkpoint restoration
    list_of_inputs.append({'export_dir': model_dir, 'tags': [], 'options': tf.saved_model.LoadOptions(experimental_skip_checkpoint=True)})
    
    # Input 3: Disallow partial checkpoint restore
    list_of_inputs.append({'export_dir': model_dir, 'tags': [], 'options': tf.saved_model.LoadOptions(allow_partial_checkpoint_restore=False)})
    
    # Input 4: Specify IO device
    list_of_inputs.append({'export_dir': model_dir, 'tags': [], 'options': tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')})
    
    # Input 5: Different variable policy
    list_of_inputs.append({'export_dir': model_dir, 'tags': [], 'options': tf.saved_model.LoadOptions(experimental_variable_policy=tf.saved_model.experimental.VariablePolicy.SAVE_AND_RESTORE_DECLARED_VARIABLES)})
    
    # Input 6: Default LoadOptions on second dir
    list_of_inputs.append({'export_dir': model_dir_2, 'tags': [], 'options': tf.saved_model.LoadOptions()})
    
    # Input 7: Skip checkpoint on second dir
    list_of_inputs.append({'export_dir': model_dir_2, 'tags': [], 'options': tf.saved_model.LoadOptions(experimental_skip_checkpoint=True)})
    
    # Input 8: Both options changed
    list_of_inputs.append({'export_dir': model_dir, 'tags': [], 'options': tf.saved_model.LoadOptions(allow_partial_checkpoint_restore=False, experimental_skip_checkpoint=True)})

    # Input 9: Another variable policy using its string name
    list_of_inputs.append({'export_dir': model_dir, 'tags': [], 'options': tf.saved_model.LoadOptions(experimental_variable_policy='checkpoint_variables')})
    
    # Input 10: Path with trailing slash
    list_of_inputs.append({'export_dir': model_dir + os.sep, 'tags': [], 'options': tf.saved_model.LoadOptions()})
    
    return list_of_inputs

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

check_valid('tf.saved_model.load', generated_inputs['tf.saved_model.load'], lib="tf", suffix=0)
