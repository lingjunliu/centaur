llvm_config=${1:-"llvm-config"}

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
if ! command -v python3.11 &> /dev/null; then
    echo "Error: python3.11 is not installed. Please install it before running this script."
    exit 1
fi
# Test coverage filtering

rm tf_test.profraw > /dev/null 2>&1
rm tf_test.profdata > /dev/null 2>&1
rm tf_test.lcov > /dev/null 2>&1

bindir=$(${llvm_config} --bindir)

if [ -z $bindir ]; then
    echo "Could not run ${llvm_config}. If LLVM is installed in a custom location, please pass the location of the llvm-config binary as an argument like this: bash filter_coverage.sh <location of llvm-config binary>"
    exit 0
else
    echo "LLVM binaries are installed in ${bindir}"
fi

libname=tensorflow
export TORCH_BUILD_DIR=$(pip show "$libname" | grep "Location:" | awk '{print $2}')/${libname}
echo "Using ${libname} from ${TORCH_BUILD_DIR}"

# generate profraw file
LLVM_PROFILE_FILE=tf_test.profraw python -c "import tensorflow as tf;print(tf.__version__)"
ls -lh tf_test.profraw

# generate profdata file
${bindir}/llvm-profdata merge -sparse tf_test.profraw -o tf_test.profdata
ls -lh tf_test.profdata

# for linux: the extension is probably .a not .dylib
LIB1=${TORCH_BUILD_DIR}/libtensorflow_cc.so.2


${bindir}/llvm-cov export -instr-profile=tf_test.profdata -format=lcov -object $LIB1 > tf_test.lcov

# check which files were covered
grep "SF:" tf_test.lcov | sort | uniq | wc -l