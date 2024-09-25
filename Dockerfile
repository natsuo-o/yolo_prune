# ベースイメージの設定
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel


# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 必要なパッケージのインストール
RUN apt update && \
    apt install -y wget curl && \
    apt install --no-install-recommends -y \
    git make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    jq libsm6 libxext6 libgl1 gfortran libatlas-base-dev liblapacke-dev vim tmux \
    libopenblas-dev liblapack-dev

# Pythonのセットアップ
RUN apt install --no-install-recommends -y python3 python3-pip python3-setuptools
RUN pip3 install --upgrade pip

# OpenCVのインストール
RUN apt install --no-install-recommends -y g++ gcc python3-dev
RUN pip3 install opencv-python==4.6.0.66

# 以下のバージョンを少しでも変更すると全く動かなくなるので注意
RUN pip3 install wandb \
    gsplat \
    ultralytics==8.2.97 \
    deepsparse==1.8.0

# 作業ディレクトリの設定
ARG WORKDIR="/app"
ENV WORKDIR=${WORKDIR}
WORKDIR ${WORKDIR}


# Install and configure OpenSSH server
# ポート番号を22から２２２２に変更
RUN apt-get update \
 && apt-get install -y openssh-server \
 && mkdir /var/run/sshd \
 && echo 'root:123' | chpasswd \
 && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's/PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config

# Set environment variables
RUN echo "export PATH=${PATH}" >> /root/.bashrc \
 && echo "export PATH=${PATH}" >> /root/.profile

# Set working directory
WORKDIR /workspace

# Expose port 2222
EXPOSE 2222

# Start the SSH server
CMD ["/usr/sbin/sshd", "-D"]