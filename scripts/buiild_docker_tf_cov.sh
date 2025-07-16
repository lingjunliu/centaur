docker build -t tf_216_instr_im . -f instrumented_tf/Dockerfile
docker run --name tf_216_instr -it tf_216_instr_im /bin/bash