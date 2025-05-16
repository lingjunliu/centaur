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

    result = torch.fft.fftn(input_tensor, s=s, dim=dim, norm=norm)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if dim is None:
        dim = list(range(len(input_tensor.shape)))
        if s is not None:
            dim = list(range(len(input_tensor.shape)))[-len(s):]

    if s is None:
        s = [input_tensor.shape[i] for i in dim]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        x = tf.cast(input_tensor, tf.complex64)
        
        if dim is None:
            dim = list(range(len(x.shape)))
        
        for d in dim:
            x = tf.signal.fft(tf.transpose(x, perm=get_transpose_permutation(x.shape, d)))
            x = tf.transpose(x, perm=get_inverse_transpose_permutation(x.shape, d))
            
        if norm == "forward":
            n = np.prod(s)
            x = x / n
        elif norm == "ortho":
            n = np.sqrt(np.prod(s))
            x = x / n

    return {"result": x.numpy()}

def get_transpose_permutation(shape, axis):
    rank = len(shape)
    perm = list(range(rank))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    return perm

def get_inverse_transpose_permutation(shape, axis):
    rank = len(shape)
    perm = list(range(rank))
    perm[-1], perm[axis] = perm[axis], perm[-1]
    return perm

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(4, 4).astype(np.float32),
        "s": (4,4),
        "dim": (0, 1),
        "norm": "backward"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()