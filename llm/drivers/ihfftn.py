import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.ihfftn(input_tensor, s=s, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", None)
        norm = input_dict.get("norm", "backward")

        input_shape = input_tensor.shape
        
        if dim is None:
            dim = list(range(len(input_shape)))
        elif isinstance(dim, int):
            dim = [dim]

        if s is None:
            s = [input_shape[d] for d in dim]
        
        fft_length = []
        for i, d in enumerate(dim):
            fft_length.append(s[i])

        def _tf_ihfftn(input_tensor, fft_length, dim, norm):
            rank = len(input_tensor.shape)
            
            if norm == "forward":
                scale = 1.0
            elif norm == "backward":
                scale = 1.0 / np.prod(fft_length)
            elif norm == "ortho":
                scale = 1.0 / np.sqrt(np.prod(fft_length))
            else:
                raise ValueError(f"Invalid norm mode: {norm}")

            shifted_dims = [(d if d >= 0 else rank + d) for d in dim]
            
            temp_tensor = tf.cast(input_tensor, tf.complex64)
            for axis in shifted_dims[::-1]:
                n = input_tensor.shape[axis]

                if axis == shifted_dims[-1]:
                    temp_tensor = tf.signal.irfft(temp_tensor, fft_length=fft_length[-1] if isinstance(fft_length[-1], list) else [fft_length[-1]])
                    temp_tensor = tf.transpose(temp_tensor, perm=get_transpose_permutation(len(temp_tensor.shape), axis))
                else:
                    temp_tensor = tf.signal.ifft(temp_tensor)
                    temp_tensor = tf.transpose(temp_tensor, perm=get_transpose_permutation(len(temp_tensor.shape), axis))
            
            for axis in shifted_dims:
                temp_tensor = tf.transpose(temp_tensor, perm=get_inverse_transpose_permutation(len(temp_tensor.shape), axis))

            return tf.cast(temp_tensor * scale, dtype=input_tensor.dtype)

        def get_transpose_permutation(rank, axis):
            permutation = list(range(rank))
            permutation.pop(axis)
            permutation.append(axis)
            return permutation

        def get_inverse_transpose_permutation(rank, axis):
            permutation = list(range(rank))
            new_axis = rank - 1
            permutation[axis] = new_axis
            permutation[new_axis] = axis
            return permutation

        real_input = tf.cast(input_tensor, dtype=tf.complex64)
        result = _tf_ihfftn(real_input, fft_length, dim, norm)
        result = tf.math.real(result).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32),
        "s": (10, 10),
        "dim": (0, 1),
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()