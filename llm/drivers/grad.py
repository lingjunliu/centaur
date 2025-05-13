import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.autograd import grad

    inputs = torch.tensor(input_dict['inputs'], requires_grad=True)
    grad_outputs = torch.tensor(input_dict['grad_outputs'])
    retain_graph = input_dict.get("retain_graph", False)
    create_graph = input_dict.get("create_graph", False)
    allow_unused = input_dict.get("allow_unused", False)

    if not cpu:
        inputs = inputs.cuda()
        grad_outputs = grad_outputs.cuda()

    result = grad(inputs, inputs, grad_outputs=grad_outputs, retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)[0]

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        inputs = tf.constant(input_dict['inputs'])
        grad_outputs = tf.constant(input_dict['grad_outputs'])
        retain_graph = input_dict.get("retain_graph", False)
        create_graph = input_dict.get("create_graph", False)
        allow_unused = input_dict.get("allow_unused", False)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            outputs = inputs * grad_outputs #Mimic grad_outputs

        result = tape.gradient(outputs, inputs)

    return {'result': result.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        'inputs': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'grad_outputs': np.array([4.0, 5.0, 6.0], dtype=np.float32),
        'retain_graph': False,
        'create_graph': False,
        'allow_unused': False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'inputs': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'grad_outputs': np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
        'retain_graph': False,
        'create_graph': False,
        'allow_unused': False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()