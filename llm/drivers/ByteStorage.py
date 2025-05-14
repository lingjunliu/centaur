import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    offset = torch.tensor(input_dict["offset"])
    index = torch.tensor(input_dict["index"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        offset = offset.cuda()
        index = index.cuda()

    buffer = input_tensor.numpy().tobytes()
    byte_storage = torch.ByteStorage.from_buffer(buffer)

    result = [byte_storage[i] for i in range(offset.item(), offset.item() + index.item())]

    if not cpu:
        pass

    return {'result': np.array(result, dtype=np.uint8)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    offset = tf.constant(input_dict["offset"])
    index = tf.constant(input_dict["index"])

    if not cpu:
        if tf.config.list_physical_devices('GPU'):
            device = '/GPU:0'
        else:
            device = '/CPU:0'
    else:
        device = '/CPU:0'

    with tf.device(device):
      input_tensor_np = input_tensor.numpy()
      offset_np = int(offset.numpy())
      index_np = int(index.numpy())

      result = input_tensor_np[offset_np:offset_np + index_np]

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint8),
        "offset": np.array(2, dtype=np.int64),
        "index": np.array(5, dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()