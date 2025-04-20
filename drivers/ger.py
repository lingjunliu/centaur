import numpy as np

def torch_version(input, cpu=True):
    import torch

    vec1 = torch.tensor(input["vec1"])
    vec2 = torch.tensor(input["vec2"])
    
    if not cpu:
        vec1 = vec1.cuda()
        vec2 = vec2.cuda()

    result = torch.outer(vec1, vec2)

    if not cpu:
        result = result.cpu()

    return {"outer_product": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        vec1 = tf.constant(input["vec1"])
        vec2 = tf.constant(input["vec2"])

        result = tf.tensordot(vec1, vec2, axes=0)

        return {"outer_product": result.numpy()}

def main():
    input_data = {
        "vec1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "vec2": np.array([4.0, 5.0, 6.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    assert np.array_equal(torch_result["outer_product"], tf_result["outer_product"]), "Results are not equal!"

    if np.array_equal(torch_result["outer_product"], tf_result["outer_product"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()