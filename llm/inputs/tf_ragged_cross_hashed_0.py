
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

class PatchedTensorList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.shape = (len(self),)
        self.dtype = tf.int32
        self.size = len(self)

def tf_ragged_cross_hashed_inputs():
    """
    Generates a list of valid inputs for the tf.ragged.cross_hashed function.
    """
    list_of_inputs = []

    # Input 1: Basic case, compatible row lengths [1, 2]
    input_dict_1 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[1], [2, 3]], dtype=tf.int64),
            tf.ragged.constant([[4], [5, 6]], dtype=tf.int64)
        ]),
        'num_buckets': 100,
        'hash_key': 1337,
        'name': 'basic_int_cross_compatible'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: No bucketing, compatible row lengths [2, 1]
    input_dict_2 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[10, 20], [30]], dtype=tf.int64),
            tf.ragged.constant([[100, 200], [300]], dtype=tf.int64)
        ]),
        'num_buckets': 0,
        'hash_key': 12345,
        'name': 'no_bucketing_compatible'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Mix of RaggedTensor and dense Tensor, compatible shapes
    input_dict_3 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[11, 22], [33, 44]], dtype=tf.int32),
            tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
        ]),
        'num_buckets': 50,
        'hash_key': -1,
        'name': 'mixed_ragged_dense_compatible'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Crossing three tensors, all compatible with row lengths [1, 2]
    input_dict_4 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[1], [2, 3]], dtype=tf.int64),
            tf.ragged.constant([[4], [5, 6]], dtype=tf.int64),
            tf.ragged.constant([[7], [8, 9]], dtype=tf.int64)
        ]),
        'num_buckets': 1000,
        'hash_key': 0,
        'name': 'three_tensors_compatible'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Inputs with empty rows, compatible row lengths [0, 1, 0, 2]
    input_dict_5 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[], [1], [], [2, 3]], dtype=tf.int64),
            tf.ragged.constant([[], [10], [], [20, 30]], dtype=tf.int64)
        ]),
        'num_buckets': 10,
        'hash_key': 999,
        'name': 'empty_rows_compatible'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Single tensor in the input list
    input_dict_6 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[101, 102], [103, 104, 105]], dtype=tf.int32)
        ]),
        'num_buckets': 200,
        'hash_key': 2024,
        'name': 'single_tensor_hash'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: All dense tensors
    input_dict_7 = {
        'inputs': PatchedTensorList([
            tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
            tf.constant([[10, 11], [12, 13]], dtype=tf.int32)
        ]),
        'num_buckets': 20,
        'hash_key': -1000,
        'name': 'all_dense_cross'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Different integer dtypes, compatible row lengths [1, 2]
    input_dict_8 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[100], [200, 300]], dtype=tf.int32),
            tf.ragged.constant([[4], [5, 6]], dtype=tf.int64)
        ]),
        'num_buckets': 0,
        'hash_key': 1,
        'name': 'mixed_dtypes_compatible'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Single row in tensors, compatible row length [3]
    input_dict_9 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[1, 2, 3]], dtype=tf.int64),
            tf.ragged.constant([[10, 20, 30]], dtype=tf.int64)
        ]),
        'num_buckets': 5,
        'hash_key': 42,
        'name': 'single_row_compatible'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensors with empty inner dimension
    input_dict_10 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[], []], dtype=tf.int64),
            tf.ragged.constant([[], []], dtype=tf.int64)
        ]),
        'num_buckets': 100,
        'hash_key': 88,
        'name': 'empty_inner_dim_cross'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Large number of buckets, compatible row lengths [4, 2]
    input_dict_11 = {
        'inputs': PatchedTensorList([
            tf.ragged.constant([[1, 2, 3, 4], [5, 6]], dtype=tf.int64),
            tf.ragged.constant([[7, 8, 9, 10], [11, 12]], dtype=tf.int64)
        ]),
        'num_buckets': 1000000,
        'hash_key': 777,
        'name': 'large_num_buckets_compatible'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.ragged.cross_hashed"] = tf_ragged_cross_hashed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.cross_hashed'.")

check_valid('tf.ragged.cross_hashed', generated_inputs['tf.ragged.cross_hashed'], lib="tf", suffix=0)
