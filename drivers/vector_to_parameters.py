import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    parameters = input_dict["parameters"]
    vector = torch.tensor(input_dict["vector"])
    
    if not cpu:
        vector = vector.cuda()
        for i in range(len(parameters)):
            parameters[i] = parameters[i].cuda()

    torch.nn.utils.vector_to_parameters(vector, parameters)
    
    if not cpu:
        for i in range(len(parameters)):
            parameters[i] = parameters[i].cpu()
    
    result = []
    for p in parameters:
        result.append(p.detach().numpy())

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    parameters = input_dict["parameters"]
    vector = tf.constant(input_dict["vector"])

    shapes = [p.shape for p in parameters]
    num_params = sum(np.prod(shape) for shape in shapes)

    if tf.size(vector) != num_params:
        raise ValueError('Number of elements in the vector should '
                         'match the number of parameters.')

    pointer = 0
    result = []

    for i, p in enumerate(parameters):
        num_param = np.prod(shapes[i])
        block = vector[pointer:pointer + num_param]
        reshaped = tf.reshape(block, shapes[i])
        
        parameters[i] = reshaped
        pointer += num_param
        result.append(reshaped.numpy())

    return {"result": result}

def main():
    A_TOL = 0.01

    example_input = {
        "parameters": [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([5.0, 6.0], dtype=np.float32)
        ],
        "vector": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    }

    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    parameters_torch = [torch.tensor(p, requires_grad=True) for p in example_input["parameters"]]
    input_torch = {"parameters": parameters_torch, "vector": example_input["vector"]}
    torch_result = torch_version(input_torch)

    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    parameters_tf = [tf.Variable(p) for p in example_input["parameters"]]
    input_tf = {"parameters": parameters_tf, "vector": example_input["vector"]}
    tf_result = tensorflow_version(input_tf)

    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()