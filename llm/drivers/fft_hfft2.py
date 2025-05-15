import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.hfft2(input_tensor, s=s, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    input_tensor = tf.constant(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    input_shape = input_tensor.shape
    dims_to_transform = dim if dim is not None else (-2, -1)

    if isinstance(dims_to_transform, int):
        dims_to_transform = [dims_to_transform]

    dims_to_transform = [d % len(input_shape) for d in dims_to_transform]
    
    s_tf = s
    if s is None:
        s_tf = [input_shape[-2], input_shape[-1]*2 - 2]
    elif isinstance(s, tuple):
        s_tf = list(s)
    elif isinstance(s, int):
        s_tf = [s]
    else:
        s_tf = [input_shape[-2], input_shape[-1]*2 - 2]

    if not isinstance(s_tf, (tuple, list)):
        raise TypeError("Shape s must be a tuple or list of ints.")

    if len(s_tf) == 0:
        s_tf = None

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        result = tf.signal.rfft2d(input_tensor, fft_length=s_tf[-2:])

        if norm == "forward":
            n = 1
            if s is None:
                n = np.prod(s_tf)
            else:
              n = np.prod(s)
            result = result / n

        elif norm == "ortho":
            n = 1
            if s is None:
                n = np.prod(s_tf)
            else:
              n = np.prod(s)
            result = result / np.sqrt(n)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 9).astype(np.float32),
        "s": (10, 10),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_res = torch_result["result"]
    tf_res = tf_result["result"]
    
    min_shape = (min(torch_res.shape[0], tf_res.shape[0]), min(torch_res.shape[1], tf_res.shape[1]))
    
    torch_cropped = torch_res[:min_shape[0], :min_shape[1]]
    tf_cropped = tf_res.real[:min_shape[0], :min_shape[1]]

    assert np.allclose(torch_cropped, tf_cropped, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()