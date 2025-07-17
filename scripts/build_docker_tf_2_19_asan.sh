docker build -t tf_219_asan_im . -f asan/tf_2_19_asan.dockerfile
docker run --name tf_219_asan -it tf_219_asan_im /bin/bash