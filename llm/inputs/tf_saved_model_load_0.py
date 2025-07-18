
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os
import tempfile


def create_saved_model_for_load(export_dir, model_type='simple', signatures=None):
    """Helper function to create and save a model."""
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    if model_type == 'simple':
        obj = tf.Module()
        obj.v = tf.Variable(5.0, dtype=tf.float32)
        obj.f = tf.function(
            lambda x: obj.v * x,
            input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
        tf.saved_model.save(obj, export_dir)
    elif model_type == 'keras':
        # Use model.export() to save a Keras model in the SavedModel format.
        inputs = tf.keras.Input(shape=(4,))
        x = tf.keras.layers.Dense(10, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.export(export_dir)
    elif model_type == 'multi_signature':
        class MultiSigModule(tf.Module):
            def __init__(self):
                self.v = tf.Variable(2.0, dtype=tf.float32)

            @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
            def multiply(self, x):
                return self.v * x

            @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
            def add(self, x):
                return self.v + x
        
        module = MultiSigModule()
        if signatures is None:
            signatures = {
                'serving_default': module.multiply,
                'add_op': module.add,
            }
        tf.saved_model.save(module, export_dir, signatures=signatures)


def tf_saved_model_load_inputs():
    list_of_inputs = []
    base_temp_dir = tempfile.mkdtemp()

    # Case 1: Simple tf.Module model with default 'serve' tag
    export_dir_1 = os.path.join(base_temp_dir, "model_1")
    create_saved_model_for_load(export_dir_1)
    input_dict_1 = {
        'export_dir': export_dir_1,
        'tags': [tf.saved_model.SERVING],
        'options': 'None'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Keras model with default 'serve' tag
    export_dir_2 = os.path.join(base_temp_dir, "model_2")
    create_saved_model_for_load(export_dir_2, model_type='keras')
    input_dict_2 = {
        'export_dir': export_dir_2,
        'tags': [tf.saved_model.SERVING],
        'options': 'experimental_io_device=/job:localhost'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Simple model, using string 'serve' for tag
    export_dir_3 = os.path.join(base_temp_dir, "model_3")
    create_saved_model_for_load(export_dir_3)
    input_dict_3 = {
        'export_dir': export_dir_3,
        'tags': ['serve'],
        'options': 'None'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Keras model, different path
    export_dir_4 = os.path.join(base_temp_dir, "model_4")
    create_saved_model_for_load(export_dir_4, model_type='keras')
    input_dict_4 = {
        'export_dir': export_dir_4,
        'tags': [tf.saved_model.SERVING],
        'options': 'None'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Simple model with a different options string
    export_dir_5 = os.path.join(base_temp_dir, "model_5")
    create_saved_model_for_load(export_dir_5)
    input_dict_5 = {
        'export_dir': export_dir_5,
        'tags': [tf.saved_model.SERVING],
        'options': 'allow_soft_placement=True'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Model in a nested directory
    export_dir_6 = os.path.join(base_temp_dir, "nested", "model_6")
    create_saved_model_for_load(export_dir_6)
    input_dict_6 = {
        'export_dir': export_dir_6,
        'tags': [tf.saved_model.SERVING],
        'options': 'None'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Model with multiple signatures
    export_dir_7 = os.path.join(base_temp_dir, "model_7")
    create_saved_model_for_load(export_dir_7, model_type='multi_signature')
    input_dict_7 = {
        'export_dir': export_dir_7,
        'tags': [tf.saved_model.SERVING],
        'options': 'None'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Simple model, different custom options string
    export_dir_8 = os.path.join(base_temp_dir, "model_8")
    create_saved_model_for_load(export_dir_8)
    input_dict_8 = {
        'export_dir': export_dir_8,
        'tags': [tf.saved_model.SERVING],
        'options': 'some_custom_string_option'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Keras model in a deeper nested directory
    export_dir_9 = os.path.join(base_temp_dir, "nested", "keras", "model_9")
    create_saved_model_for_load(export_dir_9, model_type='keras')
    input_dict_9 = {
        'export_dir': export_dir_9,
        'tags': [tf.saved_model.SERVING],
        'options': 'None'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Case 10: Model with explicitly passed signatures dictionary
    export_dir_10 = os.path.join(base_temp_dir, "model_10")
    module_for_sig = tf.Module()
    module_for_sig.f = tf.function(lambda x: x * 2.0, input_signature=[tf.TensorSpec(None, tf.float32)])
    signatures_10 = {'my_sig': module_for_sig.f}
    create_saved_model_for_load(export_dir_10, model_type='multi_signature', signatures=signatures_10)
    input_dict_10 = {
        'export_dir': export_dir_10,
        'tags': [tf.saved_model.SERVING],
        'options': 'another_option'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Case 11: Another Keras model with a different option string
    export_dir_11 = os.path.join(base_temp_dir, "model_11")
    create_saved_model_for_load(export_dir_11, model_type='keras')
    input_dict_11 = {
        'export_dir': export_dir_11,
        'tags': ['serve'],
        'options': 'experimental_skip_checkpoint=True'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

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
