import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    output_size = input_dict.get("output_size", None)
    output_ratio = input_dict.get("output_ratio", None)
    return_indices = input_dict.get("return_indices", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    pool = torch.nn.FractionalMaxPool3d(kernel_size=kernel_size, output_size=output_size,
                                         output_ratio=output_ratio, return_indices=return_indices)

    if not cpu:
        pool = pool.cuda()

    if return_indices:
        result, indices = pool(input_tensor)
        if not cpu:
            result = result.cpu()
            indices = indices.cpu()
        return {"result": result.numpy(), "indices": indices.numpy()}
    else:
        result = pool(input_tensor)
        if not cpu:
            result = result.cpu()
        return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    output_size = input_dict.get("output_size", None)
    output_ratio = input_dict.get("output_ratio", None)
    return_indices = input_dict.get("return_indices", False)

    input_shape = input_tensor.shape.as_list()

    if output_size is not None:
        target_size = output_size
    elif output_ratio is not None:
        target_size = [int(input_shape[i+1] * output_ratio[i]) for i in range(3)]
    else:
        target_size = [input_shape[i+1] // kernel_size[i] for i in range(3)]
    
    output = tf.nn.avg_pool3d(
        input=input_tensor,
        ksize=kernel_size,
        strides=kernel_size,
        padding='VALID',
    )

    return {"result": output.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 4).astype(np.float32),
        "kernel_size": (2, 2, 2),
        "output_ratio": (0.5, 0.5, 0.5),
        "return_indices": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 4).astype(np.float32),
        "kernel_size": (2, 2, 2),
        "output_size": (4,4,2),
        "return_indices": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()