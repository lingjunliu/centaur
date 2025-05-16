import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict.get("n", None)
    dim = input_dict.get("dim", -1)
    norm = input_dict.get("norm", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.fftn(input_tensor, s=n, dim=dim, norm=norm)

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
        n = input_dict.get("n", None)
        dim = input_dict.get("dim", -1)
        norm = input_dict.get("norm", None)
        
        if n is not None:
            if isinstance(n, int):
                n = [n]
            n = tf.constant(n)
        
        rank = len(input_tensor.shape)
        
        if isinstance(dim, tuple):
            dims = list(dim)
        else:
            dims = [dim]

        for i in range(len(dims)):
            if dims[i] < 0:
                dims[i] = rank + dims[i]

        if n is not None:
            shape = tf.shape(input_tensor)
            pad_shape = shape.numpy().tolist()
            
            
            if isinstance(n, tf.Tensor):
              n_val = n.numpy()
            elif isinstance(n, list):
              n_val = n
            else:
              n_val = [n]

            padding = [[0,0]] * rank
            
            for i, d in enumerate(dims):
              pad_len = max(0, n_val[i] - shape[d])

              if pad_len > 0:
                  padding[d] = [0, pad_len]

            input_tensor = tf.pad(input_tensor, padding)
        
        result = tf.cast(input_tensor, tf.complex64)

        fft_result = result
        if isinstance(dim, tuple):
          for d in sorted(dims):
            fft_result = tf.signal.fft(fft_result)
        else:
          fft_result = tf.signal.fft(fft_result)

        if norm == "ortho":
            fft_result = fft_result / tf.cast(tf.math.sqrt(tf.cast(tf.reduce_prod(tf.shape(input_tensor)), tf.float32)), tf.complex64)
            
        result = fft_result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1+1j, -1-1j], [-1+1j, 1-1j]], dtype=np.complex64),
        "n": [4,4],
        "dim": (0,1),
        "norm": "ortho"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()