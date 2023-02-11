FROM jupyter/pyspark-notebook:2021-12-21

USER root
RUN apt-get update \
    && apt install -y gcc build-essential \
    && apt-get install wget -y \
    && apt-get install curl -y \
    && apt-get install htop  -y \
    && apt-get install git -y \
    && apt-get install vim -y \
    && apt-get install llvm-8-runtime -y \
    && apt-get install postgresql libpq-dev -y \
    && conda install -c conda-forge jupyter_contrib_nbextensions \
    && pip install sparkmonitor \
    && jupyter contrib nbextension install --sys-prefix \
    && conda install -c conda-forge jupyter-resource-usage \
    && conda install -c conda-forge jupyter_nbextensions_configurator yapf\
    && jupyter nbextensions_configurator enable \
    && jupyter nbextension enable toc2/main\
    && jupyter nbextension enable code_prettify/autopep8 \
    && jupyter nbextension enable execute_time/ExecuteTime \
    && jupyter nbextension enable collapsible_headings/main \
    && jupyter nbextension enable scratchpad/main \
    && jupyter nbextension enable skip-traceback/main \
    && jupyter nbextension enable notify/notify \
    && jupyter nbextension install --py sparkmonitor --user --symlink \
    && jupyter nbextension enable sparkmonitor --user --py \
    && ipython profile create \
    && echo "c.InteractiveShellApp.extensions.append('sparkmonitor.kernelextension')" >>  $(ipython profile locate default)/ipython_kernel_config.py \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}" \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -\
   && fix-permissions "/home/${NB_USER}"

USER ${NB_USER}
COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt --quiet --no-cache-dir 
