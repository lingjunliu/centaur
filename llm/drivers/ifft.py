import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", "backward")
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.ifft(input_tensor, n=n, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        n = input_dict.get("n", None)
        dim = input_dict.get("dim", -1)
        norm = input_dict.get("norm", "backward")

        input_tensor = tf.cast(input_tensor, tf.complex128)

        if n is not None:
            input_shape = tf.shape(input_tensor)[dim if dim != -1 else -1]
            if n > input_shape:
                padding = n - input_shape
                pad_width = [(0, 0)] * len(input_tensor.shape)
                pad_width[dim] = (0, padding)
                input_tensor = tf.pad(input_tensor, pad_width)
            elif n < input_shape:
                slices = [slice(None)] * len(input_tensor.shape)
                slices[dim] = slice(0, n)
                input_tensor = input_tensor[tuple(slices)]

        result = tf.signal.ifft(input_tensor)

        if norm == "forward":
            pass
        elif norm == "backward":
            result = result / tf.cast(tf.shape(input_tensor)[dim if dim != -1 else -1], tf.complex128)
        elif norm == "ortho":
            result = result / tf.cast(tf.math.sqrt(tf.cast(tf.shape(input_tensor)[dim if dim != -1 else -1], tf.float64)), tf.complex128)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j], dtype=np.complex128)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()