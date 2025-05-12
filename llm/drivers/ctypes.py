import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    alpha = input_dict.get("alpha", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if input_dict["dtype"] == np.float64:
        torch_dtype = torch.float64
    elif input_dict["dtype"] == np.float32:
        torch_dtype = torch.float32
    elif input_dict["dtype"] == np.int64:
        torch_dtype = torch.int64
    elif input_dict["dtype"] == np.int32:
        torch_dtype = torch.int32
    else:
        raise ValueError("Unsupported PyTorch dtype")

    result = input_tensor.type(torch_dtype)

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
        alpha = input_dict.get("alpha", 1.0)

        if input_dict["dtype"] == np.float64:
          tf_dtype = tf.float64
        elif input_dict["dtype"] == np.float32:
          tf_dtype = tf.float32
        elif input_dict["dtype"] == np.int32:
          tf_dtype = tf.int32
        elif input_dict["dtype"] == np.int64:
          tf_dtype = tf.int64
        else:
          raise ValueError("Unsupported TensorFlow dtype")
    
        result = tf.cast(input_tensor, dtype=tf_dtype)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "dtype": np.float64,
        "alpha": 1
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()