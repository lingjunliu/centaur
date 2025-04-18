PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# create virtual environment (i don't want to mess up abid's code)
python -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt

(cd $PROJECT_DIR;

  python -m generator.harness

)
