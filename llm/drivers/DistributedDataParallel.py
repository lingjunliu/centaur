import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.distributed as dist

    try:
        if torch.cuda.is_available() and not cpu:
            dist.init_process_group(backend='nccl', world_size=1, init_method='tcp://127.0.0.1:23456', rank=0)
        else:
            dist.init_process_group(backend='gloo', world_size=1, init_method='tcp://127.0.0.1:23456', rank=0)
    except Exception as e:
        print(f"Warning: Could not initialize process group: {e}")
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        with torch.no_grad():
          result = module(input_tensor)
        return {"result": result.detach().numpy()}

    module = nn.Linear(10, 5)
    
    device_ids = input_dict.get("device_ids", None)
    output_device = input_dict.get("output_device", None)
    dim = input_dict.get("dim", 0)
    broadcast_buffers = input_dict.get("broadcast_buffers", True)
    process_group = input_dict.get("process_group", None)
    bucket_cap_mb = input_dict.get("bucket_cap_mb", None)
    find_unused_parameters = input_dict.get("find_unused_parameters", False)
    check_reduction = input_dict.get("check_reduction", False)
    gradient_as_bucket_view = input_dict.get("gradient_as_bucket_view", False)
    static_graph = input_dict.get("static_graph", False)
    delay_all_reduce_named_params = input_dict.get("delay_all_reduce_named_params", None)
    param_to_hook_all_reduce = input_dict.get("param_to_hook_all_reduce", None)
    mixed_precision = input_dict.get("mixed_precision", None)
    device_mesh = input_dict.get("device_mesh", None)
    
    if not cpu and torch.cuda.is_available():
        module = module.cuda()
        if device_ids is not None:
            device_ids = [torch.device("cuda", i) for i in device_ids]
        if output_device is not None:
            output_device = torch.device("cuda", output_device)


    ddp = torch.nn.parallel.DistributedDataParallel(
        module, 
        device_ids=device_ids, 
        output_device=output_device, 
        dim=dim, 
        broadcast_buffers=broadcast_buffers, 
        process_group=process_group, 
        bucket_cap_mb=bucket_cap_mb, 
        find_unused_parameters=find_unused_parameters, 
        check_reduction=check_reduction, 
        gradient_as_bucket_view=gradient_as_bucket_view, 
        static_graph=static_graph, 
        delay_all_reduce_named_params=delay_all_reduce_named_params, 
        param_to_hook_all_reduce=param_to_hook_all_reduce,
        mixed_precision=mixed_precision,
        device_mesh=device_mesh
    )

    input_tensor = torch.randn(2, 10)
    if not cpu and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    result = ddp(input_tensor)

    if not cpu and torch.cuda.is_available():
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        
        initializer = tf.keras.initializers.Ones()
        module = tf.keras.layers.Dense(5, input_shape=(10,), use_bias=False)

        device_ids = input_dict.get("device_ids", None)
        output_device = input_dict.get("output_device", None)
        dim = input_dict.get("dim", 0)
        broadcast_buffers = input_dict.get("broadcast_buffers", True)
        init_sync = input_dict.get("init_sync", True)
        process_group = input_dict.get("process_group", None)
        bucket_cap_mb = input_dict.get("bucket_cap_mb", None)
        find_unused_parameters = input_dict.get("find_unused_parameters", False)
        check_reduction = input_dict.get("check_reduction", False)
        gradient_as_bucket_view = input_dict.get("gradient_as_bucket_view", False)
        static_graph = input_dict.get("static_graph", False)
        delay_all_reduce_named_params = input_dict.get("delay_all_reduce_named_params", None)
        param_to_hook_all_reduce = input_dict.get("param_to_hook_all_reduce", None)
        mixed_precision = input_dict.get("mixed_precision", None)
        device_mesh = input_dict.get("device_mesh", None)

        input_tensor = tf.random.normal((2, 10))
        result = module(input_tensor)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_res = torch_result["result"]
    tf_res = tf_result["result"]

    assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()