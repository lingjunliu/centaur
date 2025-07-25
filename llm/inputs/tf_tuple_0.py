
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

# The API tf.tuple is designed for TFv1 graph mode.
# We disable eager execution to properly construct graph-based inputs,
# which aligns with the API's documentation and intended use case.
tf.compat.v1.disable_eager_execution()

def tf_tuple_inputs():
    list_of_inputs = []
    
    # Since this function returns symbolic tensors that are evaluated later,
    # all tensors must be created in the same graph. We reset the default graph
    # to ensure a clean state for the test run and use name_scopes to avoid
    # collisions. This setup ensures that the returned tensors remain valid
    # outside the scope of this function.
    tf.compat.v1.reset_default_graph()

    def create_tensor_object_array(tensor_list):
        arr = np.empty(len(tensor_list), dtype=object)
        for i, item in enumerate(tensor_list):
            arr[i] = item
        return arr

    # --- Input 1 ---
    with tf.compat.v1.name_scope("input_1"):
        v_1 = tf.compat.v1.Variable(0.0)
        update_op_1 = v_1.assign_add(1.0)
    tensors_1 = create_tensor_object_array([tf.constant(np.array([1.0, 2.0]), dtype=tf.float32)])
    control_inputs_1 = [update_op_1]
    input_dict_1 = {
        'tensors': tensors_1,
        'control_inputs': control_inputs_1,
        'name': 'case_1'
    }
    list_of_inputs.append(input_dict_1)

    # --- Input 2 ---
    with tf.compat.v1.name_scope("input_2"):
        v_2_1 = tf.compat.v1.Variable(10, dtype=tf.int32)
        v_2_2 = tf.compat.v1.Variable(-10.0, dtype=tf.float32)
        update_op_2_1 = v_2_1.assign_sub(1)
        update_op_2_2 = v_2_2.assign(5.0)
    tensors_2 = create_tensor_object_array([
        tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32),
        tf.constant(np.array([-5.0]), dtype=tf.float32)
    ])
    control_inputs_2 = [update_op_2_1, update_op_2_2]
    input_dict_2 = {
        'tensors': tensors_2,
        'control_inputs': control_inputs_2,
        'name': 'case_2'
    }
    list_of_inputs.append(input_dict_2)

    # --- Input 3 ---
    tensors_3 = create_tensor_object_array([tf.constant(np.array([True, False]), dtype=tf.bool)])
    input_dict_3 = {
        'tensors': tensors_3,
        'control_inputs': None,
        'name': 'case_3_no_controls'
    }
    list_of_inputs.append(input_dict_3)

    # --- Input 4 ---
    tensors_4 = create_tensor_object_array([tf.constant(np.array([b'a', b'b']), dtype=tf.string)])
    input_dict_4 = {
        'tensors': tensors_4,
        'control_inputs': [],
        'name': 'case_4_empty_controls'
    }
    list_of_inputs.append(input_dict_4)
    
    # --- Input 5 ---
    with tf.compat.v1.name_scope("input_5"):
        control_tensor_5 = tf.constant(5.0) * 2.0
    tensors_5 = create_tensor_object_array([
        tf.constant(np.array(100), dtype=tf.int64),
        tf.constant(np.array([1.0, -1.0]), dtype=tf.float64)
    ])
    control_inputs_5 = [control_tensor_5]
    input_dict_5 = {
        'tensors': tensors_5,
        'control_inputs': control_inputs_5,
        'name': 'case_5_tensor_control'
    }
    list_of_inputs.append(input_dict_5)

    # --- Input 6 ---
    with tf.compat.v1.name_scope("input_6"):
        v_6 = tf.compat.v1.Variable(0)
        update_op_6 = v_6.assign_add(1)
    tensors_6 = create_tensor_object_array([
        tf.constant(np.array([1.0]), dtype=tf.float32),
        None,
        tf.constant(np.array([3.0]), dtype=tf.float32)
    ])
    control_inputs_6 = [update_op_6]
    input_dict_6 = {
        'tensors': tensors_6,
        'control_inputs': control_inputs_6,
        'name': 'case_6_with_none'
    }
    list_of_inputs.append(input_dict_6)

    # --- Input 7 ---
    with tf.compat.v1.name_scope("input_7"):
        v_7 = tf.compat.v1.Variable(0, dtype=tf.uint8)
        update_op_7 = v_7.assign(255)
    tensors_7 = create_tensor_object_array([tf.constant(np.array([0, 0]), dtype=tf.uint8)])
    control_inputs_7 = [update_op_7]
    input_dict_7 = {
        'tensors': tensors_7,
        'control_inputs': control_inputs_7,
        'name': None
    }
    list_of_inputs.append(input_dict_7)

    # --- Input 8 ---
    with tf.compat.v1.name_scope("input_8"):
        control_placeholder_8 = tf.compat.v1.placeholder(tf.float32, shape=())
    tensors_8 = create_tensor_object_array([tf.constant(np.array([9.9]), dtype=tf.float32)])
    control_inputs_8 = [control_placeholder_8]
    input_dict_8 = {
        'tensors': tensors_8,
        'control_inputs': control_inputs_8,
        'name': 'case_8_placeholder_control'
    }
    list_of_inputs.append(input_dict_8)

    # --- Input 9 ---
    with tf.compat.v1.name_scope("input_9"):
        v_9 = tf.compat.v1.Variable(1+1j, dtype=tf.complex64)
        update_op_9 = v_9.assign(2+2j)
    tensors_9 = create_tensor_object_array([tf.constant(np.array([1+2j, 3+4j]), dtype=tf.complex128)])
    control_inputs_9 = [update_op_9]
    input_dict_9 = {
        'tensors': tensors_9,
        'control_inputs': control_inputs_9,
        'name': 'case_9_complex'
    }
    list_of_inputs.append(input_dict_9)

    # --- Input 10 ---
    with tf.compat.v1.name_scope("input_10"):
        print_op_10 = tf.print("control op for case 10")
    tensors_10 = create_tensor_object_array([tf.constant(np.zeros((1,1,1,1)), dtype=tf.float32)])
    control_inputs_10 = [print_op_10]
    input_dict_10 = {
        'tensors': tensors_10,
        'control_inputs': control_inputs_10,
        'name': 'case_10_print_op_control'
    }
    list_of_inputs.append(input_dict_10)

    return list_of_inputs

generated_inputs["tf.tuple"] = tf_tuple_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.tuple' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tuple'.")

check_valid('tf.tuple', generated_inputs['tf.tuple'], lib="tf", suffix=0)
