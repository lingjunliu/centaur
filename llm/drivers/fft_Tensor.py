import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", None)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    if s is not None:
        if dim is not None:
            result = torch.fft.fft(input_tensor, n=s[0], dim=dim, norm=norm)
        else:
            result = torch.fft.fft(input_tensor, n=s[0], norm=norm)
    elif dim is not None:
        result = torch.fft.fft(input_tensor, dim=dim, norm=norm)
    else:
        result = torch.fft.fft(input_tensor, norm=norm)

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
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", None)
        norm = input_dict.get("norm", None)
        
        rank = len(input_tensor.shape)
        if dim is None:
          dim = tuple(range(rank))
        elif isinstance(dim, int):
          dim = (dim,)

        if s is not None:
          input_shape = input_tensor.shape
          output_shape = list(input_shape)
          for d, length in zip(dim, s):
              output_shape[d] = length
          output_shape = tf.TensorShape(output_shape)
          
          padding_shape = list(input_shape)
          
          padded_input = input_tensor
          for d, length in zip(dim, s):
            if length > input_shape[d]:
              padding = [(0, 0)] * rank
              padding[d] = (0, length - input_shape[d])
              padded_input = tf.pad(padded_input, padding)
            elif length < input_shape[d]:
                slices = [slice(None)] * rank
                slices[d] = slice(0,length)
                padded_input = padded_input[tuple(slices)]
          input_tensor = padded_input

        input_tensor = tf.cast(input_tensor, tf.complex128)
        result = tf.signal.fft(input_tensor)

        if norm == "ortho":
            n = 1.0
            for d in dim:
              n *= tf.cast(tf.shape(input_tensor)[d], tf.complex128)
            result = result / tf.cast(tf.math.sqrt(tf.cast(n, tf.complex128)), tf.complex128)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "s": np.array([8], dtype = np.int32).tolist(),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "dim": 0,
        "norm": "ortho"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "norm": "ortho"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()