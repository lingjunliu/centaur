import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.fft.hfftn(input_tensor, s=s, dim=dim, norm=norm)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", None)
    norm = input_dict.get("norm", "backward")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):

        input_tensor_complex = tf.complex(tf.cast(tf.math.real(input_tensor), dtype=tf.float32), tf.cast(tf.math.imag(input_tensor), dtype=tf.float32))

        if dim is None:
            if s is not None:
                dim = list(range(len(input_tensor.shape) - len(s), len(input_tensor.shape)))
            else:
                dim = list(range(len(input_tensor.shape)))
        
        if s is not None:
            s_tf = tf.cast(s, dtype=tf.int32)
        else:
            s_tf = None

        if s_tf is None:
            s_tf = []
            for i in range(len(input_tensor.shape)):
                if i in dim:
                    if i == len(input_tensor.shape) - 1:
                        s_tf.append(2 * (input_tensor.shape[i] - 1))
                    else:
                        s_tf.append(input_tensor.shape[i])
                else:
                    s_tf.append(input_tensor.shape[i])
            s_tf = tf.constant(s_tf, dtype=tf.int32)

        result = input_tensor_complex
        for i in sorted(dim):
            result = tf.signal.fft(result)

        if norm == "forward":
            n = tf.reduce_prod(tf.cast(s_tf[:len(dim)], dtype=tf.float32))
            result = result / n
        elif norm == "ortho":
            n = tf.reduce_prod(tf.cast(s_tf[:len(dim)], dtype=tf.float32))
            result = result / tf.sqrt(n)

        result = tf.cast(tf.math.real(result), dtype=tf.float32)

        if s is not None:
            target_shape = list(s)
            current_shape = list(result.shape)
            
            while len(target_shape) < len(current_shape):
                target_shape.append(1)

            while len(current_shape) < len(target_shape):
                current_shape.append(1)

            padding = [[0, max(0, target_shape[i] - current_shape[i])] for i in range(len(target_shape))]
            result = tf.pad(result, padding)

            slices = [slice(0, target_shape[i]) for i in range(len(target_shape))]
            result = result[tuple(slices)]

            result = result.numpy()
        else:
            result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1+1j, 2+2j, 3+3j], [4+0j, 5+0j, 6+0j]], dtype=np.complex64),
        "s": (2, 2),
        "dim": (0, 1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()