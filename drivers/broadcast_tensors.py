import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input list of dictionaries
    if cpu:
        tensors = [torch.tensor(tensor) for tensor in input['tensors']]
    else:
        tensors = [torch.tensor(tensor).cuda() for tensor in input['tensors']]

    # Apply to torch.broadcast_tensors
    broadcasted_tensors = torch.broadcast_tensors(*tensors)

    if not cpu:
        broadcasted_tensors = [tensor.cpu() for tensor in broadcasted_tensors]

    return {'output': [tensor.numpy() for tensor in broadcasted_tensors]}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input list of dictionaries
        tensors = [tf.constant(tensor) for tensor in input['tensors']]

        # Apply to TensorFlow equivalent
        broadcasted_tensors = tf.broadcast_static_shape(*[tensor.shape for tensor in tensors])
        broadcasted_tensors = [tf.broadcast_to(tensor, broadcasted_tensors) for tensor in tensors]

        return {'output': [tensor.numpy() for tensor in broadcasted_tensors]}

def main():
    # Example input
    input_data = { 'tensors': [
            np.random.rand(3,),  # Random 1D tensor
            np.random.rand(3, 1)  # Random 2D tensor with broadcasting dimension
        ]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert both results to numpy arrays for comparison
    torch_np = np.array(torch_result['output'])
    tf_np = np.array(tf_result['output'])

    if np.allclose(torch_np, tf_np):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()