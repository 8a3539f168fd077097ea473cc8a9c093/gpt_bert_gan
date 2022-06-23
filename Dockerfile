# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.0-base
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update

# set it to run as non-interactive
ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# update/upgrade apt
ENV TZ=Europe/Paris
RUN apt upgrade -y

#install basics
RUN apt-get install git -yq
RUN apt-get install curl -yq

#install miniconda
ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh Miniconda3.sh
RUN bash Miniconda3.sh -b -p /miniconda
ENV PATH="/miniconda/bin:${PATH}"
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN rm Miniconda3.sh
RUN conda install wget -y

# create and activate conda environment
RUN conda create -q -n run-environement python="3.8.10" numpy scipy matplotlib
RUN /bin/bash -c "source activate run-environement"
RUN conda install python="3.8.10" pip

# install basics
RUN apt-get install less nano -yq
RUN apt-get -yq install build-essential
RUN apt-get -yq install libsuitesparse-dev
RUN apt-get -yq install wget
RUN apt-get -yq install unzip
RUN apt-get -yq install lsof
RUN apt-get update
RUN apt-get -yq install libsm6 libxrender1 libfontconfig1 libglib2.0-0

# install torch and co
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 torchsummary==1.5.1 -f https://download.pytorch.org/whl/torch_stable.html

# install huggingface
RUN pip install transformers==4.18.0
RUN pip install datasets==2.0.0

# intall nltk
RUN pip install --user -U nltk==3.6.1
RUN python -m nltk.downloader punkt

#install sklearn
RUN pip install -U scikit-learn==1.1.1

#install pandas
RUN pip install pandas==1.4.2

# check that scipy, numpy and matplotlib are well installed
RUN conda install python="3.8.10" scipy numpy matplotlib
