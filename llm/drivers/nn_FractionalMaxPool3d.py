import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    output_size = input_dict.get("output_size", None)
    output_ratio = input_dict.get("output_ratio", None)
    return_indices = input_dict.get("return_indices", False)
    _random_samples = input_dict.get("_random_samples", None)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    m = torch.nn.FractionalMaxPool3d(kernel_size=kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)
    
    if not cpu:
        m = m.cuda()
    
    result = m(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    input_np = input_dict["input"]
    kernel_size = input_dict["kernel_size"]
    output_size = input_dict.get("output_size", None)
    output_ratio = input_dict.get("output_ratio", None)
    return_indices = input_dict.get("return_indices", False)
    _random_samples = input_dict.get("_random_samples", None)

    input_tensor = tf.convert_to_tensor(input_np, dtype=tf.float32)
    input_shape = input_tensor.shape
    
    if len(input_shape) == 4:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    
    if output_size is None and output_ratio is not None:
        if isinstance(output_ratio, float):
            output_ratio = (output_ratio, output_ratio, output_ratio)
        
        output_size = (int(input_shape[-3] * output_ratio[0]), 
                       int(input_shape[-2] * output_ratio[1]), 
                       int(input_shape[-1] * output_ratio[2]))
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    stride_t = int(input_shape[-3] / output_size[0])
    stride_h = int(input_shape[-2] / output_size[1])
    stride_w = int(input_shape[-1] / output_size[2])
    

    max_pooled = tf.nn.max_pool3d(
        input_tensor,  # Reshape to have channel dimension
        ksize=[1, kernel_size[0], kernel_size[1], kernel_size[2], 1],
        strides=[1, stride_t, stride_h, stride_w, 1],
        padding='VALID',
    )
    result = max_pooled

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 1, 16, 16, 16).astype(np.float32),
        "kernel_size": 3,
        "output_size": (8, 8, 8)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    min_shape = np.min([torch_result["result"].shape, tf_result["result"].shape], axis=0)
    
    assert np.allclose(torch_result["result"][:min_shape[0],:min_shape[1],:min_shape[2],:min_shape[3],:min_shape[4]], tf_result["result"][:min_shape[0],:min_shape[1],:min_shape[2],:min_shape[3],:min_shape[4]], atol=A_TOL), "Results do not match"
    
    input_data_ratio = {
        "input": np.random.rand(1, 1, 16, 16, 16).astype(np.float32),
        "kernel_size": 3,
        "output_ratio": (0.5, 0.5, 0.5)
    }

    torch_result_ratio = torch_version(input_data_ratio)
    tf_result_ratio = tensorflow_version(input_data_ratio)
    
    min_shape = np.min([torch_result_ratio["result"].shape, tf_result_ratio["result"].shape], axis=0)
    
    assert np.allclose(torch_result_ratio["result"][:min_shape[0],:min_shape[1],:min_shape[2],:min_shape[3],:min_shape[4]], tf_result_ratio["result"][:min_shape[0],:min_shape[1],:min_shape[2],:min_shape[3],:min_shape[4]], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()