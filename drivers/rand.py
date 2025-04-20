import numpy as np

def torch_version(input, cpu=True):
    import torch

    size = input['size']
    device = 'cpu' if cpu else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        tensor = torch.rand(size, dtype=torch.float32, device=device)
    
    return {"rand_tensor": tensor.cpu().numpy()}

# Function to generate random tensors using TensorFlow
def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    size = tf.constant(input['size'], dtype=tf.int32)
    device_string = "/cpu:0" if cpu else "/gpu:0"
    
    with tf.device(device_string):
        tensor = tf.random.uniform(shape=size)
    
    return {"rand_tensor": tensor.numpy()}

def main():
    # Example input
    input_data = {
        'size': (2, 3)
    }

    # Get the torch result
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # Get the tensorflow result
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Compare the results by mean and standard deviation
    torch_mean = np.mean(torch_result["rand_tensor"])
    tf_mean = np.mean(tf_result["rand_tensor"])
    torch_std = np.std(torch_result["rand_tensor"])
    tf_std = np.std(tf_result["rand_tensor"])

    mean_close = np.isclose(torch_mean, tf_mean, atol=1e-1)
    std_close = np.isclose(torch_std, tf_std, atol=1e-1)

    if mean_close and std_close:
        print("equal")
    else:
        print("not equal")
    print(f"Torch mean: {torch_mean}, TF mean: {tf_mean}")
    print(f"Torch std: {torch_std}, TF std: {tf_std}")

if __name__ == "__main__":
    main()