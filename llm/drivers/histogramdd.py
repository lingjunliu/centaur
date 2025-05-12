import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    bins = input_dict["bins"]
    range_val = input_dict.get("range", None)
    weight = input_dict.get("weight", None)
    density = input_dict.get("density", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(bins, list):
            if all(isinstance(b, torch.Tensor) for b in bins):
                bins = tuple(b.cuda() for b in bins)
            elif all(isinstance(b, int) for b in bins):
                bins = tuple(torch.tensor(b).int() for b in bins)
            else:
                raise TypeError("bins must be a sequence of N ints, or a sequence of N 1D tensors")
        elif isinstance(bins, int):
            pass
        if weight is not None:
            weight = torch.tensor(weight)
            if not cpu:
                weight = weight.cuda()

    if isinstance(bins, list) and all(isinstance(b, int) for b in bins):
        bins = tuple(torch.tensor(b).int() for b in bins)
    elif isinstance(bins, list) and all(isinstance(b, torch.Tensor) for b in bins):
        bins = tuple(bins)
    elif isinstance(bins, int):
        pass
    else:
        raise TypeError("bins must be a sequence of N ints, or a sequence of N 1D tensors, or an int")

    if range_val is not None:
      range_val = tuple(range_val)


    hist, bin_edges = torch.histogramdd(input_tensor, bins=bins, range=range_val, weight=weight, density=density)

    if not cpu:
        hist = hist.cpu()
        bin_edges = [b.cpu() for b in bin_edges]

    return {"hist": hist.numpy(), "bin_edges": [b.numpy() for b in bin_edges]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    bins = input_dict["bins"]
    range_val = input_dict.get("range", None)
    weight = input_dict.get("weight", None)
    density = input_dict.get("density", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if isinstance(bins, int):
            if range_val is None:
                min_vals = tf.reduce_min(input_tensor, axis=0)
                max_vals = tf.reduce_max(input_tensor, axis=0)
                bin_edges = [tf.linspace(min_vals[i], max_vals[i], bins + 1) for i in range(input_tensor.shape[-1])]
            else:
                bin_edges = [tf.linspace(range_val[2*i], range_val[2*i+1], bins + 1) for i in range(input_tensor.shape[-1])]

        elif isinstance(bins, list) and all(isinstance(b, int) for b in bins):
            if range_val is None:
                min_vals = tf.reduce_min(input_tensor, axis=0)
                max_vals = tf.reduce_max(input_tensor, axis=0)
                bin_edges = [tf.linspace(min_vals[i], max_vals[i], bins[i] + 1) for i in range(input_tensor.shape[-1])]
            else:
                bin_edges = [tf.linspace(range_val[2*i], range_val[2*i+1], bins[i] + 1) for i in range(input_tensor.shape[-1])]
        
        elif isinstance(bins, list) and all(isinstance(b, tf.Tensor) for b in bins):
            bin_edges = bins
        else:
            raise TypeError("bins must be a sequence of N ints, a sequence of N 1D tensors, or an int")

        n_dims = input_tensor.shape[-1]
        n_points = input_tensor.shape[0]
        
        indices = []
        for i in range(n_dims):
            indices.append(tf.searchsorted(bin_edges[i], input_tensor[:, i], side='right') - 1)
        
        indices = tf.stack(indices, axis=-1)

        valid_bins = tf.reduce_all((indices >= 0) & (indices < tf.cast([len(be) - 1 for be in bin_edges], tf.int64)), axis=-1)
        indices = tf.boolean_mask(indices, valid_bins)
        
        if weight is not None:
            weights = tf.boolean_mask(weight, valid_bins)
        else:
            weights = tf.ones(tf.shape(indices)[0])

        
        shape = tuple(len(be) - 1 for be in bin_edges)
        
        multipliers = [1]
        for s in reversed(shape[1:]):
            multipliers.insert(0, multipliers[0] * s)
        
        multipliers = tf.constant(multipliers, dtype=tf.int64)
        
        raveled_indices = tf.reduce_sum(indices * multipliers, axis=-1)
        
        histogram = tf.scatter_nd(tf.expand_dims(raveled_indices, axis=1), weights, shape=[tf.reduce_prod(shape)])
        histogram = tf.reshape(histogram, shape)
        
        if density:
            total_weight = tf.reduce_sum(histogram)
            bin_volumes = tf.ones(shape)
            for i in range(n_dims):
                bin_widths = bin_edges[i][1:] - bin_edges[i][:-1]
                bin_volumes = tf.einsum('i,...->i...', bin_widths, bin_volumes)
            histogram = histogram / total_weight / bin_volumes

        hist = histogram.numpy()
        bin_edges = [b.numpy() for b in bin_edges]
    
    return {"hist": hist, "bin_edges": bin_edges}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0., 1.], [1., 0.], [2., 0.], [2., 2.]], dtype=np.float32),
        "bins": [3, 3],
        "weight": np.array([1., 2., 4., 8.], dtype=np.float32)
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["hist"], tf_result["hist"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32),
        "bins": [2, 2],
        "range": [0., 1., 0., 1.],
        "density": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["hist"], tf_result["hist"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()