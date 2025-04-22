import numpy as np
from utils.defaults import domain_limits, list_of_available_dtypes, MAX_SZ_TENSOR
from utils.misc import get_tensor_size

'''
    Generate a random list of lists for a domain with a random generator
    passed as an argument. For tensors, this list of list will be an
    abstact input. For other types, it will contain concrete inputs but
    still needs to be translated back.
    
    For tensors, if the generated tensor is larger than MAX_SZ_TENSOR,
    try again.
'''
def gen_ran_ll(domain, rng=np.random.default_rng(42)):
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
        return gen_ran_ll(domain, rng)
    
    return ll

'''
    Get a list of lists from a concrete input
'''
def get_ll(domain, value):
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
        list_val = list(value.shape)
        dtype_val = [list_of_available_dtypes.index(value.dtype)]
        range_val = [np.min(value), np.max(value)] if value.size > 0 else [0, 0]
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

'''
    Generate a concrete input given a list of lists. If the domain is tensor,
    the provided rng will be used to generate the concrete input.
'''
def gen_concrete_input(domain, ll, rng=np.random.default_rng(42)):
    if ll[2][0] > ll[2][1]:
        ll[2] = [ll[2][1], ll[2][0]]
    if domain in ["integer", "float", "string", "boolean", "dtype"]: # primitives and dtype
        return list_of_available_dtypes[ll[1][0]](ll[0][0])
    elif domain == "tensor": # tensors, uses the rng passed to the function
        return rng.uniform(low=ll[2][0], high=ll[2][1], size=ll[0]).astype(list_of_available_dtypes[ll[1][0]])
    elif domain == "tuple":
        return tuple([list_of_available_dtypes[ll[1][0]](x) for x in ll[0]])
    elif domain == "list":
        return [list_of_available_dtypes[ll[1][0]](x) for x in ll[0]]
    else:
        raise NotImplementedError(f"Not implemented for {domain} yet")

'''
    Generate random input according to signature and concretize it
'''    
def get_random_input(signature, rng=np.random.default_rng(42)):
    input_dict = {}
    for arg, domain in signature.items():
        # TODO: Add support for tensor_list
        if domain == "tensor_list":
            domain = "tensor"   # hack until tensor_list is supported
        
        ll = gen_ran_ll(domain, rng)    # get abstract form            
        input_dict[arg] = gen_concrete_input(domain, ll, rng) # concretize
        
    return input_dict