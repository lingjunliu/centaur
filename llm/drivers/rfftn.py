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

    result = torch.fft.rfftn(input_tensor, s=s, dim=dim, norm=norm)

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

        if s is None:
            if dim is None:
                s = input_tensor.shape
                dim = tuple(range(len(s)))
            else:
                s = [input_tensor.shape[d] for d in dim]
        
        if dim is None:
            dim = tuple(range(len(s)))

        input_tensor = tf.cast(input_tensor, tf.float32)

        result = input_tensor
        axes_list = list(dim)
        axes_list.sort()

        for axis in axes_list:
            result = tf.signal.rfft(result, fft_length=s[axis] if s else None)
            result = tf.transpose(result, perm= [i if i != axis else len(input_tensor.shape)-1 if axis < len(input_tensor.shape) else 0 for i in range(len(input_tensor.shape))])


        if norm == "forward":
            n = np.prod(s)
            result = result / n
        elif norm == "ortho":
            n = np.prod(s)
            result = result / np.sqrt(n)
        
        result = result.numpy()
    
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
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()