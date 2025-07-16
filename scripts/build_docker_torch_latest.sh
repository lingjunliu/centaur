docker build -t torch_latest_asan_im . -f asan/torch_latest_asan.dockerfile
docker run --name torch_latest_asan -it torch_latest_asan_im /bin/bash