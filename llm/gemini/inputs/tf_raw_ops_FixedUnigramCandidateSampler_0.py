
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_fixedunigramcandidatesampler_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.FixedUnigramCandidateSampler function.
    """
    list_of_inputs = []

    # Input 1: Basic case with unigrams list, non-zero seed for determinism
    input_dict_1 = {
        'true_classes': np.array([[1], [3], [5], [7]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 5,
        'unique': True,
        'range_max': 10,
        'vocab_file': "",
        'distortion': 1.0,
        'num_reserved_ids': 0,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [10., 9., 8., 7., 6., 5., 4., 3., 2., 1.],
        'seed': 1,
        'seed2': 1,
        'name': "basic_unigrams"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Non-unique sampling
    input_dict_2 = {
        'true_classes': np.array([[2], [4]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 3,
        'unique': False,
        'range_max': 5,
        'vocab_file': "",
        'distortion': 1.0,
        'num_reserved_ids': 0,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [5., 4., 3., 2., 1.],
        'seed': 1,
        'seed2': 2,
        'name': "basic_unigrams_non_unique"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: High distortion to skew probabilities
    input_dict_3 = {
        'true_classes': np.array([[0], [9]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 8,
        'unique': True,
        'range_max': 10,
        'vocab_file': "",
        'distortion': 2.5,
        'num_reserved_ids': 0,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10.0],
        'seed': 3,
        'seed2': 4,
        'name': "high_distortion"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Uniform distribution (distortion = 0.0)
    input_dict_4 = {
        'true_classes': np.array([[5], [15]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 10,
        'unique': True,
        'range_max': 20,
        'vocab_file': "",
        'distortion': 0.0,
        'num_reserved_ids': 0,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [float(i) for i in range(1, 21)],
        'seed': 42,
        'seed2': 42,
        'name': "uniform_dist"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Multiple true labels per context (num_true > 1)
    input_dict_5 = {
        'true_classes': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
        'num_true': 2,
        'num_sampled': 5,
        'unique': True,
        'range_max': 10,
        'vocab_file': "",
        'distortion': 1.0,
        'num_reserved_ids': 0,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [10., 9., 8., 7., 6., 5., 4., 3., 2., 1.],
        'seed': 5,
        'seed2': 6,
        'name': "multi_true"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: With reserved IDs
    input_dict_6 = {
        'true_classes': np.array([[2], [8]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 4,
        'unique': True,
        'range_max': 10,
        'vocab_file': "",
        'distortion': 1.0,
        'num_reserved_ids': 2,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [1., 2., 3., 4., 5., 6., 7., 8.],
        'seed': 7,
        'seed2': 8,
        'name': "reserved_ids"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Replaced sharded case - Non-unique oversampling
    input_dict_7 = {
        'true_classes': np.array([[1], [3]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 20,
        'unique': False,
        'range_max': 10,
        'vocab_file': "",
        'distortion': 1.0,
        'num_reserved_ids': 0,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [1.0] * 10,
        'seed': 77,
        'seed2': 88,
        'name': "non_unique_oversample"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))


    # Input 8: Unigrams with reserved IDs and distortion
    input_dict_8 = {
        'true_classes': np.array([[3, 4], [6, 7]], dtype=np.int64),
        'num_true': 2,
        'num_sampled': 2,
        'unique': False,
        'range_max': 8,
        'vocab_file': "",
        'distortion': 0.5,
        'num_reserved_ids': 3,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [5., 4., 3., 2., 1.],
        'seed': 11,
        'seed2': 12,
        'name': "unigrams_with_options"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large batch size
    input_dict_9 = {
        'true_classes': np.arange(10, dtype=np.int64).reshape(10, 1),
        'num_true': 1,
        'num_sampled': 10,
        'unique': True,
        'range_max': 20,
        'vocab_file': "",
        'distortion': 1.0,
        'num_reserved_ids': 0,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [float(i) for i in reversed(range(20))],
        'seed': 100,
        'seed2': 200,
        'name': "large_batch"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Non-unique sampling, multiple true classes, and reserved IDs
    input_dict_10 = {
        'true_classes': np.array([[10, 20], [30, 40]], dtype=np.int64),
        'num_true': 2,
        'num_sampled': 15,
        'unique': False,
        'range_max': 100,
        'vocab_file': "",
        'distortion': 1.5,
        'num_reserved_ids': 10,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [float(90 - i) for i in range(90)],
        'seed': 123,
        'seed2': 456,
        'name': "complex_case"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Replaced sharded case - Small range_max
    input_dict_11 = {
        'true_classes': np.array([[1]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 5,
        'unique': False,
        'range_max': 2,
        'vocab_file': "",
        'distortion': 0.75,
        'num_reserved_ids': 0,
        'num_shards': 1,
        'shard': 0,
        'unigrams': [10., 1.],
        'seed': 99,
        'seed2': 111,
        'name': "small_range"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.FixedUnigramCandidateSampler"] = tf_raw_ops_fixedunigramcandidatesampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FixedUnigramCandidateSampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FixedUnigramCandidateSampler'.")

check_valid('tf.raw_ops.FixedUnigramCandidateSampler', generated_inputs['tf.raw_ops.FixedUnigramCandidateSampler'], lib="tf", suffix=0)
