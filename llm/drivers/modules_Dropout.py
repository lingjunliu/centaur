import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", True)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.Dropout(p=p, inplace=inplace)
    m.train(training)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", True)

    if training:
        output = tf.nn.dropout(input_tensor, rate=p)
    else:
        output = input_tensor

    return {"result": output.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(100).astype(np.float32),
        "p": 0.5,
        "training": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()