
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_fixed_unigram_candidate_sampler_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 10
    vocab_file = ''
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = list(np.arange(1, range_max + 1))
    seed = 123
    name = "sampler_1"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[0, 1, 2, 3]], dtype=np.int64)
    num_true = 4
    num_sampled = 8
    unique = False
    range_max = 20
    vocab_file = ''
    distortion = 0.5
    num_reserved_ids = 1
    num_shards = 2
    shard = 1
    unigrams = list(np.arange(1, range_max + 1))
    seed = 456
    name = "sampler_2"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[5]], dtype=np.int64)
    num_true = 1
    num_sampled = 2
    unique = True
    range_max = 100
    vocab_file = ''
    distortion = 0.0
    num_reserved_ids = 2
    num_shards = 1
    shard = 0
    unigrams = list(np.random.rand(range_max))
    seed = 789
    name = "sampler_3"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    true_classes = np.array([[10, 11, 12]], dtype=np.int64)
    num_true = 3
    num_sampled = 6
    unique = False
    range_max = 50
    vocab_file = ''
    distortion = 2.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = list(np.arange(1, range_max + 1))
    seed = 101
    name = "sampler_4"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    true_classes = np.array([[0]], dtype=np.int64)
    num_true = 1
    num_sampled = 1
    unique = True
    range_max = 2
    vocab_file = ''
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [1.0, 1.0]
    seed = 202
    name = "sampler_5"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Multiple batches
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    num_true = 2
    num_sampled = 3
    unique = True
    range_max = 10
    vocab_file = ''
    distortion = 0.75
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = list(np.arange(1, range_max + 1))
    seed = 303
    name = "sampler_6"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger range_max
    true_classes = np.array([[150, 200]], dtype=np.int64)
    num_true = 2
    num_sampled = 5
    unique = False
    range_max = 500
    vocab_file = ''
    distortion = 1.2
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = list(np.arange(1, range_max + 1))
    seed = 404
    name = "sampler_7"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: num_reserved_ids > 0
    true_classes = np.array([[2, 3]], dtype=np.int64)
    num_true = 2
    num_sampled = 3
    unique = True
    range_max = 10
    vocab_file = ''
    distortion = 0.9
    num_reserved_ids = 2
    num_shards = 1
    shard = 0
    unigrams = list(np.arange(num_reserved_ids + 1, range_max + 1))
    seed = 505
    name = "sampler_8"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: shard and num_shards
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = False
    range_max = 20
    vocab_file = ''
    distortion = 0.6
    num_reserved_ids = 0
    num_shards = 4
    shard = 2
    unigrams = list(np.arange(1, range_max + 1))
    seed = 606
    name = "sampler_9"
    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    num_true = 5
    num_sampled = 10
    unique = True
    range_max = 100
    vocab_file = ''
    distortion = 1.5
    num_reserved_ids = 10
    num_shards = 1
    shard = 0
    unigrams = list(np.arange(num_reserved_ids + 1, range_max + 1))
    seed = 707
    name = "sampler_10"

    input_dict = {
        "true_classes": true_classes,
        "num_true": num_true,
        "num_sampled": num_sampled,
        "unique": unique,
        "range_max": range_max,
        "vocab_file": vocab_file,
        "distortion": distortion,
        "num_reserved_ids": num_reserved_ids,
        "num_shards": num_shards,
        "shard": shard,
        "unigrams": unigrams,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.fixed_unigram_candidate_sampler"] = tf_random_fixed_unigram_candidate_sampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.fixed_unigram_candidate_sampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.fixed_unigram_candidate_sampler'.")

check_valid('tf.random.fixed_unigram_candidate_sampler', generated_inputs['tf.random.fixed_unigram_candidate_sampler'], lib="tf", suffix=0)
