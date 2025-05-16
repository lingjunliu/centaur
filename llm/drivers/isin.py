import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    elements = torch.tensor(input_dict["elements"])
    test_elements = torch.tensor(input_dict["test_elements"])
    assume_unique = input_dict.get("assume_unique", False)
    invert = input_dict.get("invert", False)

    if not cpu:
        elements = elements.cuda()
        test_elements = test_elements.cuda()

    result = torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        elements = tf.constant(input_dict["elements"])
        test_elements = tf.constant(input_dict["test_elements"])
        assume_unique = input_dict.get("assume_unique", False)
        invert = input_dict.get("invert", False)

        elements_expanded = tf.expand_dims(elements, axis=-1)
        test_elements_expanded = tf.expand_dims(test_elements, axis=0)

        equal = tf.equal(elements_expanded, test_elements_expanded)
        result = tf.reduce_any(equal, axis=-1)

        if invert:
            result = tf.logical_not(result)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "elements": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "test_elements": np.array([2, 3], dtype=np.int32),
        "assume_unique": False,
        "invert": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "elements": np.array([1, 2, 3], dtype=np.int32),
        "test_elements": np.array(2, dtype=np.int32),
        "assume_unique": False,
        "invert": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "elements": np.array([1, 2, 3], dtype=np.int32),
        "test_elements": np.array([2,3,4], dtype=np.int32),
        "assume_unique": False,
        "invert": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()