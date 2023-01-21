# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel 开发(驱动)时太大了，也不需要
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

# XXX: IN CASE apt-get update fail, you can uncomment below two lines
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list  && \
	sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN echo [global]'\n'index-url = https://mirrors.aliyun.com/pypi/simple/ > /etc/pip.con

RUN apt-get update && apt-get install -y --no-install-recommends \
	# we have found python3.7 in base docker
	python3-pip \
	python3-setuptools \
	build-essential \
	&& \
	apt-get clean && \
	python -m pip install --upgrade pip

WORKDIR /workspace
COPY ./   /workspace

# install nnUNet !
RUN pip install -e .

CMD ["bash"]
