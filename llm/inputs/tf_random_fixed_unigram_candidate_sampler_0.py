
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
    unigrams = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seed = 123
    name = 'sampler1'

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
    num_sampled = 2
    unique = False
    range_max = 5
    vocab_file = ''
    distortion = 0.5
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [10, 5, 3, 2, 1]
    seed = 456
    name = 'sampler2'

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
    true_classes = np.array([[1, 2, 3, 4]], dtype=np.int64)
    num_true = 4
    num_sampled = 8
    unique = True
    range_max = 20
    vocab_file = ''
    distortion = 0.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [i+1 for i in range(20)]
    seed = 789
    name = 'sampler3'

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
    true_classes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    num_true = 3
    num_sampled = 6
    unique = False
    range_max = 15
    vocab_file = ''
    distortion = 2.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [i+1 for i in range(15)]
    seed = 1011
    name = 'sampler4'

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
    true_classes = np.array([[0, 1]], dtype=np.int64)
    num_true = 2
    num_sampled = 3
    unique = True
    range_max = 7
    vocab_file = ''
    distortion = 0.75
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [7, 6, 5, 4, 3, 2, 1]
    seed = 1213
    name = 'sampler5'

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
    true_classes = np.array([[2]], dtype=np.int64)
    num_true = 1
    num_sampled = 1
    unique = False
    range_max = 3
    vocab_file = ''
    distortion = 1.5
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [10, 20, 30][:3]
    seed = 1415
    name = 'sampler6'

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
    true_classes = np.array([[0, 1, 2]], dtype=np.int64)
    num_true = 3
    num_sampled = 5
    unique = True
    range_max = 12
    vocab_file = ''
    distortion = 0.2
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [10 + i for i in range(12)]
    seed = 1617
    name = 'sampler7'

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
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = False
    range_max = 10
    vocab_file = ''
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [1]*10
    seed = 123
    name = 'sampler8'

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
    unigrams = [1,2,3,4,5,6,7,8,9,10]
    seed = 123
    name = 'sampler9'

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
    true_classes = np.array([[0,1,2,3,4,5]], dtype=np.int64)
    num_true = 6
    num_sampled = 8
    unique = True
    range_max = 15
    vocab_file = ''
    distortion = 0.3
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    seed = 42
    name = 'sampler10'

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
