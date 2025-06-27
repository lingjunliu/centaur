import numpy as np
import copy
import time

from generator.rules import rule_to_distance
from utils.defaults import *
from utils.misc import get_tensor_size, has_time
from .definitions import *
from .input_generators import gen_concrete_input, get_ll

fresh_dim_len = 5 # constant define length of a new dimension to a tensor
max_num_generations = 100

num_offspring_add_sub = 70
num_offspring_size = 30

############### configuration ################

# eventually everything should be classes
class Configuration:
    def __init__(self, definition, seed):
        self.signature = definition["signature"]
        self.ruleset = definition["ruleset"]
        self.random_candidate = definition["random_candidate"]
        self.arg_order = definition["arg_order"]
        self.limits = definition["limits"]
        self.rng = np.random.default_rng(seed)
    
    ## set the random candidate
    def set_random_candidate(self, random_candidate):
        self.random_candidate = random_candidate
    
    ## give me a random candidate
    def get_random_candidate(self):
        return self.random_candidate

    # tell me how how to obtain a list of list of numbers from the input
    # arguments. the evolutionary algorithm only understand (and can
    # mutate) list of numbers.
    def translate(self, arg_dict):
        ll = []
        for arg in self.arg_order:
            ll += get_ll(self.signature[arg], arg_dict[arg])
        return ll
    
    def translate_back(self, args, input, abstract=False, rng=None):
        if rng is None:
            rng = self.rng
        translated_list = []
        for arg in args:
            value_ind, dtype_ind, range_ind = self.arg_order.index(arg)*3, self.arg_order.index(arg)*3 + 1, self.arg_order.index(arg)*3 + 2
            ll = [input[value_ind], input[dtype_ind], input[range_ind]]
            if abstract:
                if self.signature[arg] == "tensor": # tensors                
                    translated_list.append({arg: f"shape: {tuple(input[value_ind])}, dtype: {list_of_available_dtypes[input[dtype_ind][0]]}, range: {tuple(input[range_ind])}"})
                else:
                    translated_list.append({arg: f"value: {input[value_ind][0]}, dtype: {list_of_available_dtypes[input[dtype_ind][0]]}"})
            else:
                translated_list.append({arg: gen_concrete_input(self.signature[arg], ll, arg=arg, rng=rng)})

        return translated_list
    
    # Check if the abstrations will generate oversized tensors
    def oversized(self, args, input):
        for arg in args:
            value_ind, dtype_ind, range_ind = self.arg_order.index(arg)*3, self.arg_order.index(arg)*3 + 1, self.arg_order.index(arg)*3 + 2
            ll = [input[value_ind], input[dtype_ind], input[range_ind]]
            if get_tensor_size(ll) > MAX_SZ_TENSOR:
                return True
        return False
    
    def translate_to_input_dict(self, input, seed=None, abstract=False):
        # Allow seed to be passed for concretization
        if seed is None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        input_dict = {}
        input_list = self.translate_back(self.arg_order, input, abstract=abstract, rng=rng)
        for entry in input_list:
            for key, value in entry.items():
                input_dict[key] = value
        return input_dict
            
    
    # measuring distance
    def distance(self, input):
        dist = 0
        for arity, rule_name, *args in self.ruleset:
            # add the distance
            # if input contains oversized tensor, add max distance to
            # discourage oversized tensors
            dist += rule_to_distance[arity][rule_name](*self.translate_back(args, input)) if not self.oversized(args, input) else 1
        
        # return arithmatic mean of the distances
        return dist/len(self.ruleset)

############### mutation ################

class Mutator:
    def __init__(self, config):
        self.rng = config.rng  # Random generator ensures reproducibility
        self.config = config
        self.limits = config.limits
        # indices for limits
        self.min_number = 0
        self.max_number = 1
        self.min_size = 2
        self.max_size = 3
    
    def mutate_numbers(self, ll):
        newl = copy.deepcopy(ll)
        # Randomly pick a list
        random_choice = self.rng.integers(len(newl))
        # Get limits for this list
        limits = self.limits[random_choice]
        # index into ll
        alist = newl[random_choice]
        # don't mutate empty lists
        if len(alist) == 0:
            return newl
        random_choice = self.rng.integers(len(alist))
        if self.rng.choice([True, False]):
            if alist[random_choice] == limits[self.max_number]:
                return newl
            alist[random_choice] += 1
        else:
            if alist[random_choice] == limits[self.min_number]:
                return newl
            alist[random_choice] -= 1
        return newl

    def mutate_sizes(self, ll):
        newl = copy.deepcopy(ll)
        # Randomly pick a list
        random_choice = self.rng.integers(len(newl))
        # Get limits for this list
        limits = self.limits[random_choice]
        # index into ll
        alist = newl[random_choice]
        if self.rng.choice([True, False]):
            # remove
            if len(alist) == limits[self.min_size]:
                return newl
            # Select one of the dimensions
            random_choice = self.rng.integers(len(alist))
            del alist[random_choice]
        else:
            # add
            # Select one of the dimensions
            if len(alist) == limits[self.max_size]:
                return newl
            random_choice = self.rng.integers(len(alist)) if len(alist) > 0 else 0
            alist.insert(random_choice, self.rng.integers(fresh_dim_len))
        return newl    
    
    # mutator is a higher-order function: either mutate_numbers or mutate_sizes
    def mutate(self, best_distance, input, mutator):
        tmp_value = mutator(input)
        dist = self.config.distance(tmp_value)
        if dist < best_distance:
            return (dist, tmp_value)
        return (best_distance, input)

############### optimization ################

def optimize(config, mutator, duration=0):
    input = config.translate(config.get_random_candidate())
    best_input = input
    best_distance = config.distance(input)
    # Mutators
    mutation_limit = {
        mutator.mutate_numbers: num_offspring_add_sub,
        mutator.mutate_sizes: num_offspring_size
    }
    iteration_limit = sum(mutation_limit.values())
    start = time.time()
    for x in range(max_num_generations):
        # Counters to keep track of each mutator
        mutation_counter = {
            mutator.mutate_numbers: 0,
            mutator.mutate_sizes: 0
        }
        # Round robin
        while sum(mutation_counter.values()) < iteration_limit and has_time(start, duration):
            for f_mutator, counter in mutation_counter.items():
                if counter < mutation_limit[f_mutator]:
                    mutation_counter[f_mutator] += 1
                    (best_distance, best_input) = mutator.mutate(best_distance, best_input, f_mutator)
                    if best_distance == 0:  # great
                        return (best_distance, best_input)        
    # if got to this point, it could not find a solution. Sorry!
    return (best_distance, best_input) 
