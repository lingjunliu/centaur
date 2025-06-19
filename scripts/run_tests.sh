PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# create virtual environment
if ! command -v python3.12 &> /dev/null; then
    echo "Error: python3.12 is not installed. Please install it before running this script."
    exit 1
fi
python3.12 -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt

(cd $PROJECT_DIR; pytest)
