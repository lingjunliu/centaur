import numpy as np
from utils.defaults import domain_limits_torch, domain_limits_tf, list_of_available_dtypes, MAX_SZ_TENSOR, list_of_string_values_torch, list_of_string_values_tf
from utils.misc import get_tensor_size

def gen_ran_ll(domain, rng=np.random.default_rng(42), lib="torch"):
    '''
        Generate a random list of lists for a domain with a random generator
        passed as an argument. For tensors, this list of list will be an
        abstact input. For other types, it will contain concrete inputs but
        still needs to be translated back.
        
        For tensors, if the generated tensor is larger than MAX_SZ_TENSOR,
        try again.
    '''
    domain_limits = domain_limits_torch if lib == "torch" else domain_limits_tf
    if domain not in domain_limits or f'{domain}_dtype' not in domain_limits or f'{domain}_value_range' not in domain_limits:
        raise NotImplementedError(f"Limits not implemented for {domain}")
    
    ll = []
    limits = [
        domain_limits[domain],
        domain_limits[f'{domain}_dtype'],
        domain_limits[f'{domain}_value_range']
    ]
    
    for limit in limits:
        sz = rng.integers(limit[2], limit[3], endpoint=True)
        l = []
        for _ in range(sz):
            l.append(rng.integers(limit[0], limit[1], endpoint=True))
        ll.append(l)
        
    if domain == "tensor" and get_tensor_size(ll) > MAX_SZ_TENSOR:   # Too large, try again
        return gen_ran_ll(domain, rng, lib=lib)
    # String domain generates indices into the curated vocabulary, not raw
    # values — map indices back to the actual string tokens (per library).
    if domain == "string":
        string_list = list_of_string_values_torch if lib == "torch" else list_of_string_values_tf
        ll[0] = [string_list[ll[0][0]]]
        ll[2] = [string_list[ll[2][0]], string_list[ll[2][1]]]

    return ll

def get_ll(domain, value):
    '''
        Get a list of lists from a concrete input
    '''
    if domain == "tensor_list":
        domain = "tensor"   # hack until tensor_list is supported
    
    ll = []
    if domain in ["integer", "float", "string", "boolean", "dtype"]: # primitives and dtype
        list_val = [value]
        dtype_val = [list_of_available_dtypes.index(np.dtype(type(value)))]
        range_val = [value, value]  # for cohesion, not really needed
        # the extra np.dtype call is needed because python primitive data types are not on the list
        # and putting them on the list confuses the distance function
    elif domain == "tensor": # tensors
        if value is None:
            list_val = []
            dtype_val = [list_of_available_dtypes.index(np.float64)]  # default dtype
            range_val = [None, None]
        else:
            list_val = list(value.shape)
            dtype_val = [list_of_available_dtypes.index(value.dtype)]
            if isinstance(value, np.ndarray):
                range_val = [np.min(value), np.max(value)] if value.size > 0 else [0, 0]
            else:
                try:
                    range_val = [np.min(value), np.max(value)] if value.size > 0 else [0, 0]
                except Exception as e:
                    range_val = [0, 0]  # if the tensor is empty, set range to 0, 0
    elif domain in ["tuple", "list"]:
        list_val = list(value)
        dtype_val = [list_of_available_dtypes.index(np.dtype(type(value[0])))] if len(value) > 0 else [list_of_available_dtypes.index(np.int64)]
        range_val = [np.min(value), np.max(value)] if len(value) > 0 else [0, 0]
    else:
        raise NotImplementedError(f"Not implemented for the domain of {domain} yet")
    
    ll.append(list_val)
    ll.append(dtype_val)
    ll.append(range_val)
    return ll

def get_abstract_input(concrete, signature):
    '''
        Given a concrete input and the signature, return a dictionary that
        contains the abstract input instead of concrete inputs
    '''
    if 'args' in signature.keys() and 'kwargs' in signature.keys():
        arg_part = get_abstract_input(concrete, signature['args'])
        kwarg_part = get_abstract_input(concrete, signature['kwargs'])
        abstract = {**arg_part, **kwarg_part}
        return abstract
    abstract = {}
    for arg, domain in signature.items():
        abstract[arg] = get_ll(domain, concrete[arg])
    
    return abstract

def gen_concrete_input(domain, ll, arg="", rng=np.random.default_rng(42)):
    '''
        Generate a concrete input given a list of lists. If the domain is tensor,
        the provided rng will be used to generate the concrete input.
    '''    
    if domain in ["integer", "float", "string", "boolean", "dtype"]: # primitives and dtype
        return list_of_available_dtypes[ll[1][0]](ll[0][0]) if ll[0][0] is not None else None
    elif domain == "tensor" or domain == "tensor_list": # tensors, uses the rng passed to the function        
        # Check high > low
        if ll[2][0] > ll[2][1]: # swap them
            ll[2] = [ll[2][1], ll[2][0]]
        
        # Check if range is finite and valid
        highest_limit = np.finfo(np.float64).max
        if list_of_available_dtypes[ll[1][0]] != bool and not np.isfinite(ll[2][1] - ll[2][0]):
            # clip extremes
            if ll[2][0] < -highest_limit:
                ll[2][0] = -highest_limit
            if ll[2][1] > highest_limit:
                ll[2][1] = highest_limit
            if not np.isfinite(ll[2][1] - ll[2][0]):
                # Still not finite, so we need to adjust the range
                if -1*ll[2][0] > ll[2][1]:  # low is extreme, preserve that
                    ll[2][1] = ll[2][0] + highest_limit
                else:   # high is extreme, preserve that
                    ll[2][0] = ll[2][1] - highest_limit
        elif list_of_available_dtypes[ll[1][0]] == bool:
            if not np.isfinite(ll[2][0]):
                ll[2][0] = 1
            if not np.isfinite(ll[2][1]):
                ll[2][1] = 1
            # adjust ranges with modulo 2 for bools
            ll[2] = [ll[2][0]%2, ll[2][1]%2]
            if ll[2][0] > ll[2][1]: # swap them
                ll[2] = [ll[2][1], ll[2][0]]

        val = rng.uniform(low=ll[2][0], high=ll[2][1], size=ll[0]) if None not in ll[0] else None
        return val.astype(list_of_available_dtypes[ll[1][0]]) if val is not None else None
    elif domain == "tuple":
        # CORNER CASE: If the arg is out, the tuple is a tuple of tensors
        if arg == "out":
            return tuple([np.array([]) for x in ll[0]])
        
        return tuple([list_of_available_dtypes[ll[1][0]](x) if x is not None else None for x in ll[0]])
    elif domain == "list":
        return [list_of_available_dtypes[ll[1][0]](x) if x is not None else None for x in ll[0]]
    else:
        raise NotImplementedError(f"Not implemented for {domain} yet")
    
def concretize_input(abstract, signature, rng=np.random.default_rng(42)):
    '''
        Given an abstract input dictionary, the signature and the random
        generator to generate the original inputs, recreate the concrete
        input.
    '''
    concrete = {}
    i = 0
    for arg, domain in signature.items():
        if isinstance(abstract, dict):
            ll = abstract[arg]
        else:   # if abstract is a list
            ll = [abstract[i], abstract[i+1], abstract[i+2]]
            i += 3
        concrete[arg] = gen_concrete_input(domain, ll, arg=arg, rng=rng)
        
    return concrete

def abstract_print(abstract, signature):
    '''
        Given an abstract input dictionary and the signature and get the
        abstract input in a human readable format.
    '''
    if 'args' in signature.keys() and 'kwargs' in signature.keys():
        arg_part = abstract_print(abstract, signature['args'])
        kwarg_part = abstract_print(abstract, signature['kwargs'])
        return arg_part + kwarg_part
    printable = ""
    i = 0
    for arg, domain in signature.items():
        if isinstance(abstract, dict):
            ll = abstract[arg]
        else:   # if abstract is a list
            ll = [abstract[i], abstract[i+1], abstract[i+2]]
            i += 3
        
        if domain == "tensor": # tensors                
            printable += f'{arg}: \n\tshape: {tuple(ll[0])}\n\tdtype: {list_of_available_dtypes[ll[1][0]]}\n\trange: {(np.format_float_positional(ll[2][0]), np.format_float_positional(ll[2][1]))}\n'
        elif domain == "float": # floats
            printable += f'{arg}: \n\tvalue: {np.format_float_positional(ll[0][0]) if ll[0][0] is not None else None}\n\tdtype: {list_of_available_dtypes[ll[1][0]]}\n'
        else:
            printable += f'{arg}: \n\tvalue: {ll[0][0]}\n\tdtype: {list_of_available_dtypes[ll[1][0]]}\n'
        
    return printable

def get_random_input(signature, rng=np.random.default_rng(42), lib="torch"):
    '''
        Generate random input according to signature and concretize it
    '''
    input_dict = {}
    abstract_inp = {}
    for arg, domain in signature.items():
        # TODO: Add support for tensor_list
        if domain == "tensor_list":
            domain = "tensor"   # hack until tensor_list is supported
        
        ll = gen_ran_ll(domain, rng, lib=lib)    # get abstract form
        abstract_inp[arg] = ll          # save abstract input
        input_dict[arg] = gen_concrete_input(domain, ll, arg=arg, rng=rng) # concretize
        
    return input_dict, abstract_inp