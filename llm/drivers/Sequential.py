import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    layers = input_dict["layers"]
    
    if not cpu:
        for i in range(len(layers)):
            if isinstance(layers[i], torch.nn.Module):
                layers[i] = layers[i].cuda()

    model = torch.nn.Sequential(*layers)

    input_tensor = torch.tensor(input_dict["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = model(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        layers = input_dict["layers"]
        input_tensor = tf.constant(input_dict["input"])
        
        x = input_tensor
        for layer in layers:
            if isinstance(layer, tf.keras.layers.Dense):
                x = layer(x)
            elif isinstance(layer, tf.keras.layers.ReLU):
                x = layer(x)
            elif isinstance(layer, tf.keras.layers.Flatten):
                x = layer(x)
            elif isinstance(layer, tf.keras.layers.Conv2D):
                x = layer(x)
            elif isinstance(layer, tf.keras.layers.MaxPool2D):
                x = layer(x)
            elif isinstance(layer, tf.keras.layers.Softmax):
                x = layer(x)
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
        
        result = x.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_shape = (28, 28, 1)
    num_classes = 10
    
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    input_data = {
        "input": np.random.rand(1, 28, 28, 1).astype(np.float32),
        "layers": [
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
            tf.keras.layers.Softmax()
        ]
    }

    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    class Flatten(torch.nn.Module):
        def forward(self, input):
            return input.reshape(input.size(0), -1)

    torch_input_data = {
        "input": np.transpose(input_data["input"], (0, 3, 1, 2)),
        "layers": [
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False),
            Flatten(),
            torch.nn.Linear(13 * 13 * 32, num_classes),
            torch.nn.Softmax(dim=1)
        ]
    }

    torch_result = torch_version(torch_input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()