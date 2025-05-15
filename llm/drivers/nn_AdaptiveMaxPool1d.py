import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]
    return_indices = input_dict.get("return_indices", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    m = torch.nn.AdaptiveMaxPool1d(output_size, return_indices=return_indices)
    output = m(input_tensor)

    if not cpu:
        output = output.cpu()
    
    if return_indices:
      return {"result": output[0].numpy(), "indices": output[1].numpy()}
    else:
      return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        output_size = input_dict["output_size"]
        return_indices = input_dict.get("return_indices", False)

        input_shape = input_tensor.shape
        if len(input_shape) == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        in_len = int(input_tensor.shape[-1])

        pooled_values = []
        indices = []

        for i in range(output_size):
            start = int(i * in_len / output_size)
            end = int((i + 1) * in_len / output_size)

            window = input_tensor[:, :, start:end]
            max_val = tf.reduce_max(window, axis=2, keepdims=True)
            index = tf.argmax(window, axis=2, output_type=tf.int32) + start
            
            pooled_values.append(max_val)
            indices.append(tf.expand_dims(tf.cast(index, tf.float32), axis=1))

        result = tf.concat(pooled_values, axis=2)
        final_indices = tf.concat(indices, axis=1)

        if len(input_shape) == 2:
            result = tf.squeeze(result, axis=0)
            final_indices = tf.squeeze(final_indices, axis=0)
        
        result = tf.transpose(result, perm=[0, 2, 1])
        final_indices = tf.transpose(final_indices, perm=[0, 2, 1])

        if return_indices:
          return {"result": result.numpy(), "indices": final_indices.numpy()}
        else:
          return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 64, 8).astype(np.float32),
        "output_size": 5,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data_with_indices = {
        "input": np.random.rand(1, 64, 8).astype(np.float32),
        "output_size": 5,
        "return_indices": True,
    }

    torch_result_with_indices = torch_version(input_data_with_indices)
    tf_result_with_indices = tensorflow_version(input_data_with_indices)

    assert np.allclose(torch_result_with_indices["result"], tf_result_with_indices["result"], atol=A_TOL), "Results with indices do not match"
    assert np.allclose(torch_result_with_indices["indices"], tf_result_with_indices["indices"], atol=A_TOL), "Indices do not match"

    print("Success")

if __name__ == "__main__":
    main()