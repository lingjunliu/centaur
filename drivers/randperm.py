import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    n = input["n"]
    generator = input.get("generator", None)
    dtype = input.get("dtype", torch.int64)
    device = torch.device("cpu") if cpu else torch.device("cuda")
    requires_grad = input.get("requires_grad", False)

    # Apply to torch.randperm
    result = torch.randperm(n, generator=generator, dtype=dtype, device=device, requires_grad=requires_grad)

    if not cpu:
        result = result.cpu()

    return {"randperm": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        n = input["n"]
        
        # TensorFlow equivalent does not have a direct randperm function, we need to shuffle a range
        r = tf.range(n, dtype=tf.int64)
        result = tf.random.shuffle(r)

        return {"randperm": result.numpy()}

def main():
    # Example input
    input_data = {
        "n": 10,
        "generator": None,
        "dtype": torch.int64,
        "requires_grad": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # For comparing
    np_torch_result = torch_result["randperm"]
    np_tf_result = tf_result["randperm"]

    # Sort the results to compare since we are dealing with permutations
    if np.array_equal(np.sort(np_torch_result), np.sort(np_tf_result)):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()