
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
    unigrams = [1] * 10
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
    true_classes = np.array([[0]], dtype=np.int64)
    num_true = 1
    num_sampled = 5
    unique = False
    range_max = 20
    vocab_file = ''
    distortion = 0.5
    num_reserved_ids = 2
    num_shards = 2
    shard = 1
    unigrams = [1] * 20
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
    true_classes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    num_true = 3
    num_sampled = 6
    unique = True
    range_max = 15
    vocab_file = ''
    distortion = 1.5
    num_reserved_ids = 1
    num_shards = 1
    shard = 0
    unigrams = [1] * 15
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
    true_classes = np.array([7], dtype=np.int64)
    num_true = 1
    num_sampled = 8
    unique = False
    range_max = 30
    vocab_file = ''
    distortion = 0.0
    num_reserved_ids = 0
    num_shards = 3
    shard = 2
    unigrams = [1] * 30
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
    true_classes = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    num_true = 2
    num_sampled = 3
    unique = True
    range_max = 12
    vocab_file = ''
    distortion = 0.75
    num_reserved_ids = 3
    num_shards = 1
    shard = 0
    unigrams = [1] * 12
    seed = 222
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

     # Input 6
    true_classes = np.array([[]], dtype=np.int64)
    num_true = 0
    num_sampled = 1
    unique = True
    range_max = 1
    vocab_file = ''
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [1.0]
    seed = 123
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

    # Input 7
    true_classes = np.array([[1, 2, 3, 4]], dtype=np.int64)
    num_true = 4
    num_sampled = 7
    unique = False
    range_max = 25
    vocab_file = ''
    distortion = 0.25
    num_reserved_ids = 4
    num_shards = 5
    shard = 3
    unigrams = [1] * 25
    seed = 456
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

    # Input 8
    true_classes = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
    num_true = 5
    num_sampled = 9
    unique = True
    range_max = 18
    vocab_file = ''
    distortion = 1.75
    num_reserved_ids = 2
    num_shards = 2
    shard = 0
    unigrams = [1]*18
    seed = 789
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

    # Input 9
    true_classes = np.array([0,1,2], dtype=np.int64)
    true_classes = np.expand_dims(true_classes, axis=0)
    num_true = 3
    num_sampled = 10
    unique = False
    range_max = 3
    vocab_file = ''
    distortion = 0.5
    num_reserved_ids = 0
    num_shards = 4
    shard = 1
    unigrams = [1] * 3
    seed = 101
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
    true_classes = np.array([[1,2],[3,4],[5,6],[7,8]], dtype=np.int64)
    num_true = 2
    num_sampled = 12
    unique = True
    range_max = 9
    vocab_file = ''
    distortion = 1.25
    num_reserved_ids = 0
    num_shards = 3
    shard = 2
    unigrams = [1] * 9
    seed = 222
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

    # Input 11
    true_classes = np.array([[1]], dtype=np.int64)
    num_true = 1
    num_sampled = 2
    unique = True
    range_max = 11
    vocab_file = ''
    distortion = 0.5
    num_reserved_ids = 1
    num_shards = 1
    shard = 0
    unigrams = [1] * 11
    seed = 333
    name = "sampler_11"

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

    # Input 12
    true_classes = np.array([[0, 1, 2, 3]], dtype=np.int64)
    num_true = 4
    num_sampled = 6
    unique = True
    range_max = 4
    vocab_file = ''
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [1] * 4
    seed = 444
    name = "sampler_12"

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

    # Input 13
    true_classes = np.array([[1,2,3,4,5,6,7,8]], dtype=np.int64)
    num_true = 8
    num_sampled = 16
    unique = False
    range_max = 32
    vocab_file = ''
    distortion = 0.75
    num_reserved_ids = 0
    num_shards = 2
    shard = 1
    unigrams = [1] * 32
    seed = 555
    name = "sampler_13"

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
