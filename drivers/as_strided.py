import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Convert input to torch tensor
    input_tensor = torch.tensor(input['input'])
    size = input['size']
    stride = input['stride']
    storage_offset = input.get('storage_offset', 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply as_strided in PyTorch
    strided_tensor = torch.as_strided(input_tensor, size, stride, storage_offset)

    if not cpu:
        strided_tensor = strided_tensor.cpu()

    return {"as_strided_result": strided_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input['input'])
        size = input['size']
        stride = input['stride']
        storage_offset = input.get('storage_offset', 0)

        def tf_as_strided(x, size, stride, storage_offset):
            flat_tensor = tf.reshape(x, [-1])
            start_idx = storage_offset 
            idxs = []
            for i in range(size[0]):
                for j in range(size[1]):
                    idxs.append(start_idx + i * stride[0] + j * stride[1])

            strided_tensor = tf.gather(flat_tensor, idxs)
            strided_tensor = tf.reshape(strided_tensor, size)
            return strided_tensor

        strided_tensor = tf_as_strided(input_tensor, size, stride, storage_offset)

        return {"as_strided_result": strided_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.1, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8],
                           [0.9, 1.0, 1.1, 1.2]], dtype=np.float32),
        "size": [2, 2],
        "stride": [1, 2],
        "storage_offset": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.array_equal(torch_result["as_strided_result"], tf_result["as_strided_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()