#!/bin/bash
CONDA_ROOT=$HOME/miniconda3
CONDA=${CONDA_ROOT}/bin/conda

if [[ -z "{!PROJECT_ROOT}" ]]; then
    echo "'PROJECT_ROOT' is not set. Check that the submit file contains the line 'environment = PROJECT_HOME=\$ENV(PWD)'"
    exit 1
else
    echo "'PROJECT_ROOT=$PROJECT_ROOT'"
fi

# Check if environment exists
if [ ! -f ${CONDA} ]; then
  echo "miniconda3 is not installed. Run condor_submit setup.sub first!"
  exit 0
fi

ENV_FILE=$PROJECT_ROOT/environment.yml
ENV_NAME=$(awk -F ': ' '/name:/ {print $2}' $ENV_FILE)

echo "Running 'run.sh' in conda env $ENV_NAME"

cd ${PROJECT_ROOT}
${CONDA} run -n ${ENV_NAME} bash ${PROJECT_ROOT}/run.sh

