PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt

(cd $PROJECT_DIR;

  PYTHONWARNINGS="ignore" python -m generator.harness ${1:-0}

)
