import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    import torch

    dtype = input_dict.get("dtype", torch.float32)

    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    result = torch.tensor([0], dtype=torch.float32 if dtype == np.float32 else dtype).to(device).numpy()
    if not cpu:
      result = np.array(result)
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    dtype = input_dict.get("dtype", tf.float32)

    with tf.device('/CPU:0' if cpu else '/GPU:0'):
        result = tf.constant([0], dtype=dtype).numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "dtype": np.float32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()