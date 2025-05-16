import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    params = input_dict['params']

    for name, param in params.items():
        params[name] = torch.tensor(param)
        if not cpu:
            params[name] = params[name].cuda()

    lr = input_dict.get("lr", 1e-3)
    momentum = input_dict.get("momentum", 0)
    dampening = input_dict.get("dampening", 0)
    weight_decay = input_dict.get("weight_decay", 0)
    nesterov = input_dict.get("nesterov", False)
    maximize = input_dict.get("maximize", False)
    foreach = input_dict.get("foreach", None)
    differentiable = input_dict.get("differentiable", False)

    optimizer = torch.optim.SGD(params.values(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, maximize=maximize, foreach=foreach, differentiable=differentiable)

    for name, param in params.items():
        param.grad = torch.tensor(input_dict['grads'][name])
        if not cpu:
            param.grad = param.grad.cuda()

    optimizer.step()

    results = {}
    for name, param in params.items():
        if not cpu:
            param = param.cpu()
        results[name] = param.detach().numpy()
    
    return results

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    params = input_dict['params']

    with tf.device(device_string):
        for name, param in params.items():
            params[name] = tf.Variable(param)

        lr = input_dict.get("lr", 1e-3)
        momentum = input_dict.get("momentum", 0)
        weight_decay = input_dict.get("weight_decay", 0)

        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, weight_decay=weight_decay)

        grads = {}
        for name in params.keys():
          grads[name] = tf.constant(input_dict['grads'][name])

        variables = list(params.values())
        gradients = [grads[name] for name in params.keys()]

        optimizer.apply_gradients(zip(gradients, variables))

        results = {}
        for name, param in params.items():
            results[name] = param.numpy()
    
    return results

def main():
    A_TOL = 0.01

    params_data = {
        'param1': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'param2': np.array([4.0, 5.0, 6.0], dtype=np.float32)
    }

    grads_data = {
        'param1': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'param2': np.array([0.4, 0.5, 0.6], dtype=np.float32)
    }

    input_data = {
        "params": params_data,
        "grads": grads_data,
        "lr": 0.01,
        "momentum": 0.9
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for name in params_data.keys():
        assert np.allclose(torch_result[name], tf_result[name], atol=A_TOL), f"Results do not match for {name}"

    print("Success")

if __name__ == "__main__":
    main()