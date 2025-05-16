import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    groups = input_dict["groups"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.native_channel_shuffle(input_tensor, groups)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    groups = input_dict["groups"]

    shape = tf.shape(input_tensor)
    num_dims = tf.rank(input_tensor)

    if num_dims == 4:
      batchsize, num_channels, height, width = shape[0], shape[1], shape[2], shape[3]
      channels_per_group = num_channels // groups

      input_tensor = tf.reshape(
          input_tensor,
          [batchsize, groups, channels_per_group, height, width])
      input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3, 4])
      output_tensor = tf.reshape(
          input_tensor,
          [batchsize, channels_per_group * groups, height, width])
    elif num_dims == 3:
      num_channels, height, width = shape[0], shape[1], shape[2]
      channels_per_group = num_channels // groups

      input_tensor = tf.reshape(
          input_tensor,
          [groups, channels_per_group, height, width])
      input_tensor = tf.transpose(input_tensor, [1, 0, 2, 3])
      output_tensor = tf.reshape(
          input_tensor,
          [channels_per_group * groups, height, width])
    else:
      num_channels = shape[0]
      channels_per_group = num_channels // groups
      input_tensor = tf.reshape(
          input_tensor,
          [groups, channels_per_group])
      input_tensor = tf.transpose(input_tensor, [1, 0])
      output_tensor = tf.reshape(
          input_tensor,
          [channels_per_group * groups])

    result = output_tensor.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32),
        "groups": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()