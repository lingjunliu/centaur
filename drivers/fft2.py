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
    
    result = torch.fft.fft2(input_tensor, s=s, dim=dim, norm=norm)
    
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
        dim = input_dict.get("dim", (-2, -1))
        norm = input_dict.get("norm", "backward")

        input_rank = len(input_tensor.shape)
        axes = [input_rank + d if d < 0 else d for d in dim]

        fft_result = tf.cast(input_tensor, tf.complex64)
        
        original_shape = tf.shape(input_tensor)

        if s is not None:
          pad_sizes = []
          for i, axis in enumerate(axes):
              pad_len = s[i] - original_shape[axis]
              if pad_len > 0:
                  pad_sizes.append([0, pad_len])
              elif pad_len < 0:
                  fft_result = tf.slice(fft_result, begin=[0 if k != axis else 0 for k in range(input_rank)], size=[original_shape[k] if k != axis else s[i] for k in range(input_rank)])
                  pad_sizes.append([0, 0])
              else:
                pad_sizes.append([0, 0])

          paddings = [[0,0]] * input_rank
          for i, axis in enumerate(axes):
            paddings[axis] = pad_sizes[i]
            
          fft_result = tf.pad(fft_result, paddings)
          

        for axis in reversed(axes):
            fft_result = tf.signal.fft(fft_result)
        
        n = 1.0
        if norm == "forward":
            if s is not None:
                for i in s:
                    n *= i
            else:
                for axis in axes:
                    n *= tf.cast(tf.shape(input_tensor)[axis], tf.float32)
            fft_result = fft_result / tf.cast(n, tf.complex64)
        elif norm == "ortho":
            if s is not None:
                for i in s:
                    n *= i
            else:
                for axis in axes:
                    n *= tf.cast(tf.shape(input_tensor)[axis], tf.float32)
            fft_result = fft_result / tf.cast(tf.sqrt(n), tf.complex64)

        result = fft_result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(8, 8).astype(np.float32),
        "s": (6, 6)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()