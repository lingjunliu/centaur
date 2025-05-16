import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    sorted_sequence = torch.tensor(input["sorted_sequence"])
    values = torch.tensor(input["values"])
    out_int32 = input.get("out_int32", False)
    right = input.get("right", False)

    if not cpu:
        sorted_sequence = sorted_sequence.cuda()
        values = values.cuda()
    
    result = torch.searchsorted(
        sorted_sequence, values, 
        out_int32=out_int32, 
        right=right,
    )

    if not cpu:
        result = result.cpu()

    return {"searchsorted_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    sorted_sequence = tf.constant(input["sorted_sequence"])
    values = tf.constant(input["values"])
    right = input.get("right", False)

    with tf.device(device_string):
        if right:
            result = tf.searchsorted(sorted_sequence, values, side='right', out_type=tf.int32 if input.get("out_int32", False) else tf.int64)
        else:
            result = tf.searchsorted(sorted_sequence, values, side='left', out_type=tf.int32 if input.get("out_int32", False) else tf.int64)

    return {"searchsorted_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "sorted_sequence": [1, 3, 5, 7, 9],
        "values": [3, 6, 9],
        "out_int32": False,
        "right": False,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.array_equal(torch_result["searchsorted_result"], tf_result["searchsorted_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()