
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def fixed_unigram_candidate_sampler_inputs():
    list_of_inputs = []

    # Input 1
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 10
    vocab_file = ""
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seed = 1
    seed2 = 1
    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    true_classes = np.array([[1, 2, 3]], dtype=np.int64)
    num_true = 3
    num_sampled = 5
    unique = False
    range_max = 3
    vocab_file = ""
    distortion = 0.5
    num_reserved_ids = 1
    num_shards = 2
    shard = 1
    unigrams = [0.1, 0.2, 0.3]
    seed = 123
    seed2 = 456
    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    true_classes = np.array([[1]], dtype=np.int64)
    num_true = 1
    num_sampled = 2
    unique = True
    range_max = 5
    vocab_file = ""
    distortion = 2.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [0.1, 0.2, 0.3, 0.4, 0.5]
    seed = 789
    seed2 = 101
    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: multiple batches
    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    num_true = 2
    num_sampled = 3
    unique = False
    range_max = 8
    vocab_file = ""
    distortion = 0.0
    num_reserved_ids = 2
    num_shards = 1
    shard = 0
    unigrams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    seed = 1
    seed2 = 1
    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: different unigram distribution
    true_classes = np.array([[0, 1]], dtype=np.int64)
    num_true = 2
    num_sampled = 3
    unique = True
    range_max = 5
    vocab_file = ""
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [0.4, 0.3, 0.2, 0.1, 0.0]
    seed = 1
    seed2 = 2
    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Larger range_max
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 100
    vocab_file = ""
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [0.1] * 100
    seed = 1
    seed2 = 1
    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: num_shards > 1
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 10
    vocab_file = ""
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 4
    shard = 2
    unigrams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seed = 1
    seed2 = 1
    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: num_reserved_ids > 0
    true_classes = np.array([[1, 2]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 10
    vocab_file = ""
    distortion = 1.0
    num_reserved_ids = 3
    num_shards = 1
    shard = 0
    unigrams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seed = 1
    seed2 = 1

    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    true_classes = np.array([[1000, 2000]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 5000
    vocab_file = ""
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [0.1] * 5000
    seed = 1
    seed2 = 1
    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    true_classes = np.array([[1000, 2000], [3000, 4000]], dtype=np.int64)
    num_true = 2
    num_sampled = 4
    unique = True
    range_max = 5000
    vocab_file = ""
    distortion = 1.0
    num_reserved_ids = 0
    num_shards = 1
    shard = 0
    unigrams = [0.1] * 5000
    seed = 1
    seed2 = 1
    name = None

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
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FixedUnigramCandidateSampler"] = fixed_unigram_candidate_sampler_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FixedUnigramCandidateSampler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FixedUnigramCandidateSampler'.")

check_valid('tf.raw_ops.FixedUnigramCandidateSampler', generated_inputs['tf.raw_ops.FixedUnigramCandidateSampler'], lib="tf", suffix=0)
