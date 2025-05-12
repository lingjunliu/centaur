import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    bins = input_dict["bins"]
    range_val = input_dict.get("range", None)
    weight = input_dict.get("weight", None)
    density = input_dict.get("density", False)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    if weight is not None:
        weight = torch.tensor(weight)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(bins, torch.Tensor):
            bins = bins.cuda()
        if weight is not None:
            weight = weight.cuda()
    
    if isinstance(bins, int):
        result, bin_edges = torch.histogram(input_tensor, bins=bins, range=range_val, weight=weight, density=density)
    else:
        result, bin_edges = torch.histogram(input_tensor, bins=bins, weight=weight, density=density)
    
    if not cpu:
        result = result.cpu()
        bin_edges = bin_edges.cpu()
    
    return {"result": result.numpy(), "bin_edges": bin_edges.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        bins = input_dict["bins"]
        range_val = input_dict.get("range", None)
        weight = input_dict.get("weight", None)
        density = input_dict.get("density", False)

        if isinstance(bins, np.ndarray):
            bins = tf.constant(bins)

        if weight is not None:
            weight = tf.constant(weight)

        if isinstance(bins, int):
            if range_val is None:
                min_val = tf.reduce_min(input_tensor)
                max_val = tf.reduce_max(input_tensor)
            else:
                min_val = range_val[0]
                max_val = range_val[1]

            bin_edges = tf.linspace(tf.cast(min_val, tf.float32), tf.cast(max_val, tf.float32), bins + 1)

            counts = tf.zeros(bins, dtype=tf.float32)

            for i in range(bins):
                lower_bound = bin_edges[i]
                upper_bound = bin_edges[i+1]
                mask = tf.logical_and(input_tensor >= lower_bound, input_tensor < upper_bound)
                
                if weight is None:
                    bin_count = tf.reduce_sum(tf.cast(mask, tf.float32))
                else:
                    bin_count = tf.reduce_sum(tf.where(mask, tf.cast(weight, tf.float32), tf.zeros_like(weight, dtype=tf.float32)))

                counts = tf.tensor_scatter_nd_update(counts, [[i]], [bin_count])
        else:
            bin_edges = bins
            bins = len(bin_edges) - 1
            counts = tf.zeros(bins, dtype=tf.float32)

            for i in range(bins):
                lower_bound = bin_edges[i]
                upper_bound = bin_edges[i+1]
                mask = tf.logical_and(input_tensor >= lower_bound, input_tensor < upper_bound)

                if weight is None:
                    bin_count = tf.reduce_sum(tf.cast(mask, tf.float32))
                else:
                    bin_count = tf.reduce_sum(tf.where(mask, tf.cast(weight, tf.float32), tf.zeros_like(weight, dtype=tf.float32)))

                counts = tf.tensor_scatter_nd_update(counts, [[i]], [bin_count])

        if density:
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            area = tf.reduce_sum(counts * bin_widths)
            counts = counts / area

        result = counts.numpy()
        bin_edges = bin_edges.numpy()
    
    return {"result": result, "bin_edges": bin_edges}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1., 2, 1], dtype=np.float32),
        "bins": 4,
        "range": (0., 3.),
        "weight": np.array([1., 2., 4.], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match (counts)"
    assert np.allclose(torch_result["bin_edges"], tf_result["bin_edges"], atol=A_TOL), "Results do not match (bin_edges)"

    input_data = {
        "input": np.array([1., 2, 1], dtype=np.float32),
        "bins": 4,
        "range": (0., 3.),
        "weight": np.array([1., 2., 4.], dtype=np.float32),
        "density": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match (density)"
    assert np.allclose(torch_result["bin_edges"], tf_result["bin_edges"], atol=A_TOL), "Results do not match (bin_edges)"

    input_data = {
        "input": np.array([1., 2, 1, 5, 1, 0.5], dtype=np.float32),
        "bins": np.array([0., 1., 2., 3., 4., 5., 6.], dtype=np.float32),
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match (bins as tensor)"
    assert np.allclose(torch_result["bin_edges"], tf_result["bin_edges"], atol=A_TOL), "Results do not match (bin_edges)"

    print("Success")

if __name__ == "__main__":
    main()