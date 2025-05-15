import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.autograd import grad

    inputs = torch.tensor(input_dict["inputs"], requires_grad=True)
    grad_outputs = torch.tensor(input_dict["grad_outputs"])
    retain_graph = input_dict.get("retain_graph", False)
    create_graph = input_dict.get("create_graph", False)
    only_inputs = input_dict.get("only_inputs", True)
    allow_unused = input_dict.get("allow_unused", False)

    if not cpu:
        inputs = inputs.cuda()
        grad_outputs = grad_outputs.cuda()

    result = grad(inputs, inputs, grad_outputs=grad_outputs, retain_graph=retain_graph, create_graph=create_graph, only_inputs=only_inputs, allow_unused=allow_unused)

    if not cpu:
        if result is not None:
            if isinstance(result, tuple):
                result = tuple(r.cpu() if r is not None else None for r in result)
            else:
                result = result.cpu()

    if result is None:
        return {'result': None}
    if isinstance(result, tuple):
        return {'result': tuple(r.numpy() if r is not None else None for r in result)}
    else:
        return {'result': result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        inputs = tf.Variable(input_dict["inputs"], dtype=tf.float32)
        grad_outputs = tf.constant(input_dict["grad_outputs"], dtype=tf.float32)
        retain_graph = input_dict.get("retain_graph", False)
        create_graph = input_dict.get("create_graph", False)
        only_inputs = input_dict.get("only_inputs", True)
        allow_unused = input_dict.get("allow_unused", False)

        with tf.GradientTape(persistent=create_graph) as tape:
            loss = tf.reduce_sum(inputs * grad_outputs)

        result = tape.gradient(loss, inputs)

        if result is None:
            return {'result': None}
        else:
            result = result.numpy()

        return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "inputs": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "grad_outputs": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "retain_graph": False,
        "create_graph": False,
        "only_inputs": True,
        "allow_unused": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if torch_result['result'] is None and tf_result['result'] is None:
        print("Both results are None, considering them equal")
    elif torch_result['result'] is not None and tf_result['result'] is not None:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    else:
        assert False, "One result is None while the other is not"

    print("Success")

if __name__ == "__main__":
    main()