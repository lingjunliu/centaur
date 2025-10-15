# Collect coverage using Docker (Put resource limits here)
export max_parallel=${1:-100}         # Maximum number of parallel jobs (set this based on the number of slurm jobs you want to spawn to run at the same time)
export max_memory_docker=${2:-400G}   # Maximum memory for Docker container for TensorFlow Coverage (set this based on the memory you want to allocate for Docker)

docker build -t tf_216_instr_im . -f instrumented_tf/Dockerfile
docker run --memory=${max_memory_docker} --cpus=${max_parallel} --cpuset-cpus="0-$((${max_parallel}-1))" --name tf_216_instr -it tf_216_instr_im bash