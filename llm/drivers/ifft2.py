import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

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

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", (-2, -1))
        norm = input_dict.get("norm", "backward")

        input_shape = input_tensor.shape
        rank = len(input_shape)
        dims_to_transform = [d % rank for d in dim]

        if s is not None:
            target_shape = list(input_shape)
            for i, d in enumerate(dims_to_transform):
                if s[i] != -1:
                    target_shape[d] = s[i]
            
            pad_width = []
            for i in range(rank):
                if i in dims_to_transform:
                    idx = dims_to_transform.index(i)
                    if s[idx] > input_shape[i]:
                        pad_before = 0
                        pad_after = s[idx] - input_shape[i]
                        pad_width.append((pad_before, pad_after))
                    elif s[idx] < input_shape[i]:
                        pad_before = 0
                        pad_after = 0
                        pad_width.append((pad_before, pad_after))
                    else:
                        pad_width.append((0, 0))
                else:
                    pad_width.append((0, 0))

            input_tensor = tf.pad(input_tensor, pad_width)
        
            slices = []
            for i in range(rank):
                if i in dims_to_transform:
                    idx = dims_to_transform.index(i)
                    if s is not None and s[idx] != -1:
                        slices.append(slice(0, s[idx]))
                    else:
                        slices.append(slice(None))
                else:
                    slices.append(slice(None))
            input_tensor = input_tensor[tuple(slices)]
        
        axes = [d % len(input_tensor.shape) for d in dim]
        result = tf.cast(input_tensor, tf.complex128)

        for axis in reversed(axes):
            result = tf.signal.ifft(result)
            perm = list(range(len(input_tensor.shape)))
            perm.pop(axis)
            perm.insert(0, axis)
            result = tf.transpose(result, perm=perm)
                
        perm_back = list(range(len(input_tensor.shape)))
        for i in range(len(axes)):
            current_axis = axes[i]
            perm_back = list(range(len(input_tensor.shape)))
            perm_back.pop(0)
            perm_back.insert(current_axis, 0)
            result = tf.transpose(result, perm=perm_back)
        
        if norm == "backward":
            n = 1.0
            if s is None:
                for d in dim:
                    n *= tf.cast(input_tensor.shape[d % rank], tf.float64)
            else:
                for size in s:
                    if size != -1:
                        n *= size
                    else:
                        n = float('inf')
                if n == float('inf'):
                    n = 1.0
                    for d in dim:
                        n *= tf.cast(input_shape[d % rank], tf.float64)
            result = result / tf.cast(n, tf.complex128)

        elif norm == "ortho":
            n = 1.0
            if s is None:
                for d in dim:
                    n *= tf.cast(input_tensor.shape[d % rank], tf.float64)
            else:
                for size in s:
                    if size != -1:
                        n *= size
                    else:
                        n = float('inf')
                if n == float('inf'):
                    n = 1.0
                    for d in dim:
                        n *= tf.cast(input_shape[d % rank], tf.float64)
            result = result / tf.cast(np.sqrt(n), tf.complex128)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 10).astype(np.complex64),
        "s": (5,5),
        "dim": (-2,-1),
        "norm": "backward"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()