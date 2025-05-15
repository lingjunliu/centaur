import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"])
    dim = input_dict.get("dim", -1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    result = torch.linalg.cross(input_tensor, other_tensor, dim=dim)

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
        other_tensor = tf.constant(input_dict["other"])
        dim = input_dict.get("dim", -1)

        shape = tf.shape(input_tensor)
        rank = tf.rank(input_tensor)
        dim = dim if dim >= 0 else dim + tf.get_static_value(rank)

        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        other_tensor = tf.cast(other_tensor, dtype=tf.float32)

        if dim == tf.get_static_value(rank) - 1:
          a0 = input_tensor[..., 0]
          a1 = input_tensor[..., 1]
          a2 = input_tensor[..., 2]
          b0 = other_tensor[..., 0]
          b1 = other_tensor[..., 1]
          b2 = other_tensor[..., 2]

          out0 = a1 * b2 - a2 * b1
          out1 = a2 * b0 - a0 * b2
          out2 = a0 * b1 - a1 * b0

          result = tf.stack([out0, out1, out2], axis=-1)
        else:
            
          input_tensor_list = tf.unstack(input_tensor, axis=dim)
          other_tensor_list = tf.unstack(other_tensor, axis=dim)

          a0 = input_tensor_list[0]
          a1 = input_tensor_list[1]
          a2 = input_tensor_list[2]
          b0 = other_tensor_list[0]
          b1 = other_tensor_list[1]
          b2 = other_tensor_list[2]

          out0 = a1 * b2 - a2 * b1
          out1 = a2 * b0 - a0 * b2
          out2 = a0 * b1 - a1 * b0
          result = tf.stack([out0, out1, out2], axis=0)
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[-0.3956, 1.1455, 1.6895], [-0.5849, 1.3672, 0.3599], [-1.1626, 0.7180, -0.0521], [-0.1339, 0.9902, -2.0225]], dtype=np.float32),
        "other": np.array([[-0.0257, -1.4725, -1.2251], [-1.1479, -0.7005, -1.9757], [-1.3904, 0.3726, -1.1836], [-0.9688, -0.7153, 0.2159]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()