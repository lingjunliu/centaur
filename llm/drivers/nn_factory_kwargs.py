import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    factory_kwargs = {}
    if "layout" in input_dict:
        factory_kwargs["layout"] = input_dict["layout"]
    if "device" in input_dict:
        factory_kwargs["device"] = input_dict["device"]
    if "memory_format" in input_dict:
        factory_kwargs["memory_format"] = input_dict["memory_format"]
        
    if not cpu:
        factory_kwargs["device"] = "cuda"

    result = torch.nn.factory_kwargs(**factory_kwargs) if factory_kwargs else torch.nn.factory_kwargs(kwargs={})
    
    if not cpu and "device" not in input_dict:
        try:
            result = result.cpu()
        except:
            pass

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    factory_kwargs = {}

    if "dtype" in input_dict:
        factory_kwargs["dtype"] = input_dict["dtype"]
        
    if "layout" in input_dict:
        pass

    if "device" in input_dict:
        pass

    if "requires_grad" in input_dict:
        pass

    if "memory_format" in input_dict:
        pass

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    class DummyLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(DummyLayer, self).__init__()

    with tf.device(device_string):
        layer = DummyLayer(**factory_kwargs)
        config = layer.get_config()
        return {"result": config}

def main():
    A_TOL = 0.01
    input_data = {}
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()