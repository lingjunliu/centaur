
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ReaderRestoreState_inputs():
    list_of_inputs = []

    # Input 1
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)

    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)

    state = tf.constant("state_string")

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    state = tf.constant("")

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    state = tf.constant("Another state string")

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    state = tf.constant("State with some numbers 1234567890")

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    state = tf.constant("State with special characters !@#$%^&*()")

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    state = tf.constant("State with unicode characters 你好世界")

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    state = tf.constant("Very long state " * 200)

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    state = tf.constant("State with mixed characters 123abc你好")

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    state = tf.constant("\n\t\r\f")

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    try:
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        reader_handle = iterator.string_handle()
        with tf.compat.v1.Session() as sess:
            reader_handle_value = sess.run(reader_handle)
        reader_handle = tf.constant(reader_handle_value)
    state = tf.constant("This is a basic state string.")

    input_dict = {
        "reader_handle": reader_handle,
        "state": state,
        "name": "restore_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReaderRestoreState"] = tf_raw_ops_ReaderRestoreState_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReaderRestoreState' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderRestoreState'.")

check_valid('tf.raw_ops.ReaderRestoreState', generated_inputs['tf.raw_ops.ReaderRestoreState'], lib="tf", suffix=0)
