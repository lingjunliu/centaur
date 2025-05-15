import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.rfft2(input_tensor, s=s, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    rank = len(input_tensor.shape)
    axes = [(rank + d) % rank for d in dim]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):

        if s is not None:
            input_shape = tf.shape(input_tensor)
            padding = []
            target_shape = []
            for i, axis in enumerate(axes):
                if s[i] == -1:
                    padding.append([0,0])
                    target_shape.append(input_shape[axis])
                elif s[i] > input_shape[axis]:
                    padding_before = 0
                    padding_after = s[i] - input_shape[axis]
                    padding.append([padding_before, padding_after])
                    target_shape.append(s[i])
                elif s[i] < input_shape[axis]:
                    start = 0
                    end = s[i]
                    input_tensor = tf.slice(input_tensor, begin=[0] * rank, size=input_shape)
                    indices = []
                    for j in range(rank):
                        if j == axis:
                            indices.append(slice(0, s[i]))
                        else:
                            indices.append(slice(None))
                    input_tensor = input_tensor[tuple(indices)]
                    padding = None
                    target_shape = None
                    break

            if padding:
                paddings = [[0, 0] for _ in range(rank)]
                for i, axis in enumerate(axes):
                    paddings[axis] = padding[i]
                input_tensor = tf.pad(input_tensor, paddings)

        result = tf.signal.rfft2d(input_tensor, fft_length=s if s is not None else None)

        if norm == "forward":
            if s is None:
                sz = 1.0
                for d in dim:
                    sz *= tf.cast(tf.shape(input_tensor)[(rank+d)%rank], tf.float32)
                result = result / tf.cast(sz, dtype=result.dtype)
            else:
                sz = 1.0
                for ss in s:
                    sz *= float(ss)
                result = result / tf.cast(sz, dtype=result.dtype)
        elif norm == "ortho":
            if s is None:
                sz = 1.0
                for d in dim:
                    sz *= tf.cast(tf.shape(input_tensor)[(rank+d)%rank], tf.float32)
                result = result / tf.cast(np.sqrt(sz), dtype=result.dtype)
            else:
                sz = 1.0
                for ss in s:
                    sz *= float(ss)
                result = result / tf.cast(np.sqrt(sz), dtype=result.dtype)


        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(5, 7, 12, 15).astype(np.float32),
        "dim": (-2, -1),
        "norm": 'backward'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(5, 7, 12, 15).astype(np.float32),
        "dim": (-2, -1),
        "norm": 'forward'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(5, 7, 12, 15).astype(np.float32),
        "dim": (-2, -1),
        "norm": 'ortho'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(5, 7, 12, 15).astype(np.float32),
        "s": (12,15),
        "dim": (-2, -1),
        "norm": 'backward'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(5, 7, 12, 15).astype(np.float32),
        "s": (24,30),
        "dim": (-2, -1),
        "norm": 'backward'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()