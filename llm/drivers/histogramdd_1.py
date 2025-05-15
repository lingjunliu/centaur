import numpy as np

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
            if all(isinstance(b, int) for b in bins):
                bins = tuple(torch.tensor(b) for b in bins)
            elif all(isinstance(b, torch.Tensor) for b in bins):
                bins = [b.cuda() for b in bins]
        if weight is not None:
            weight = torch.tensor(weight)
            if not cpu:
                weight = weight.cuda()

    if isinstance(bins, list) and all(isinstance(b, int) for b in bins):
        bins = tuple(torch.tensor(b) for b in bins)
        if weight is not None:
            result = torch.histogramdd(input_tensor, bins=bins, range=range_val, weight=weight, density=density)
        else:
            result = torch.histogramdd(input_tensor, bins=bins, range=range_val, density=density)
    else:
        if weight is not None:
            result = torch.histogramdd(input_tensor, bins=bins, range=range_val, weight=weight, density=density)
        else:
            result = torch.histogramdd(input_tensor, bins=bins, range=range_val, density=density)

    if not cpu:
        hist = result.hist.cpu()
        bin_edges = tuple(edge.cpu() for edge in result.bin_edges)
    else:
        hist = result.hist
        bin_edges = result.bin_edges
    
    return {"hist": hist.numpy(), "bin_edges": [edge.numpy() for edge in bin_edges]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    bins = input_dict["bins"]
    range_val = input_dict.get("range", None)
    weight = input_dict.get("weight", None)
    density = input_dict.get("density", False)

    if isinstance(bins, int):
        num_dims = input_tensor.shape[-1]
        bins_list = [bins] * num_dims
    elif isinstance(bins, list) and all(isinstance(b, int) for b in bins):
        bins_list = bins
    else:
        raise ValueError("TensorFlow version only supports integer bins")

    if range_val is None:
        min_vals = tf.reduce_min(input_tensor, axis=0)
        max_vals = tf.reduce_max(input_tensor, axis=0)
        range_vals = []
        for i in range(input_tensor.shape[-1]):
            range_vals.extend([min_vals[i].numpy(), max_vals[i].numpy()])

    else:
        range_vals = range_val

    hist, bin_edges = tf.raw_ops.HistogramFixedWidth(
        values=input_tensor,
        value_range=range_vals,
        nbins=bins_list
    )
    
    bin_edges_list = []
    start_idx = 0
    for n_bins in bins_list:
        end_idx = start_idx + n_bins + 1
        bin_edges_list.append(tf.slice(range_vals, [start_idx], [2]).numpy())
        start_idx += 2

    hist = tf.cast(hist, dtype=tf.float32)

    if weight is not None:
        weight = tf.constant(weight, dtype=tf.float32)
        hist_shape = hist.shape
        hist = tf.zeros(hist_shape, dtype=tf.float32)
        
        input_shape = input_tensor.shape
        
        for i in range(input_shape[0]):
            point = input_tensor[i]
            w = weight[i] if len(weight.shape) > 0 else weight
            
            bin_indices = []
            for j in range(input_shape[-1]):
                bin_width = (range_vals[2*j+1] - range_vals[2*j]) / bins_list[j]
                bin_index = tf.floor((point[j] - range_vals[2*j]) / bin_width)
                bin_indices.append(tf.cast(bin_index, dtype=tf.int32))
                
            try:
                hist_idx = tuple(tf.get_static_value(tf.clip_by_value(idx, 0, bins_list[k]-1)).item() for k, idx in enumerate(bin_indices))
                hist = tf.tensor_scatter_nd_update(hist, [list(hist_idx)], [w])
            except:
                pass #Ignore index erros

    if density:
        total_count = tf.reduce_sum(hist)
        bin_volumes = 1.0
        for i in range(len(bins_list)):
            bin_width = (range_vals[2*i+1] - range_vals[2*i]) / bins_list[i]
            bin_volumes *= bin_width
        hist = hist / total_count / bin_volumes

    return {"hist": hist.numpy(), "bin_edges": bin_edges_list}

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
    print("Success")

if __name__ == "__main__":
    main()