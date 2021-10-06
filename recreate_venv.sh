HEREDIR=`pwd`
VENV_DIR=${HEREDIR}/venv
echo "Creating venv in: ${VENV_DIR}"

[ -d "${VENV_DIR}" ] && rm -rf "${VENV_DIR}";
virtualenv -p python3.7 ${VENV_DIR}
echo export PYTHONPATH="${HEREDIR}" >> ${VENV_DIR}/bin/activate
source ${VENV_DIR}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Submodule init/update.."
git submodule init
git submodule update