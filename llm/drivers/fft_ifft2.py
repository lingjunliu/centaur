import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.ifft2(input_tensor, s=s, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        rank = len(input_tensor.shape)

        if dim is None:
            dim = list(range(rank))
        elif isinstance(dim, int):
            dim = [dim]

        dim = [rank + d if d < 0 else d for d in dim]
        
        dim = sorted(dim)

        if len(dim) != 2:
          raise ValueError("tensorflow_version: dim must have length 2")

        if s is None:
          s = [input_tensor.shape[d] for d in dim]

        if s is not None:
            s = list(s)
            if len(s) != 2:
              raise ValueError("tensorflow_version: s must have length 2")

            padded_input = input_tensor
            for i,d in enumerate(dim):
                current_size = padded_input.shape[d]
                target_size = s[i]
                if target_size != -1 and target_size != current_size:
                    if target_size > current_size:
                        pad_size = target_size - current_size
                        paddings = [[0, 0]] * rank
                        paddings[d] = [0, pad_size]
                        padded_input = tf.pad(padded_input, paddings, "CONSTANT")
                    else:
                        slices = [slice(None)] * rank
                        slices[d] = slice(0, target_size)
                        padded_input = padded_input[tuple(slices)]
            input_tensor = padded_input

        input_tensor = tf.cast(input_tensor, tf.complex128)
        fft2d_result = tf.signal.fft2d(input_tensor)
        ifft2d_result = tf.signal.ifft2d(fft2d_result)
        result = tf.cast(ifft2d_result, tf.complex64)

        n = 1.0
        if s is not None:
          n = np.prod(s)
        else:
            n = np.prod([input_tensor.shape[d] for d in dim])
        
        if norm == "forward":
            pass
        elif norm == "backward":
            result = result / n
        elif norm == "ortho":
            result = result / np.sqrt(n)
        else:
            raise ValueError("Invalid norm mode: {}".format(norm))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 10).astype(np.complex64),
        "s": (10, 10),
        "dim": (-2, -1),
        "norm": "backward"
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"].flatten().real, tf_result["result"].flatten().real, atol=A_TOL)
    assert np.allclose(torch_result["result"].flatten().imag, tf_result["result"].flatten().imag, atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()