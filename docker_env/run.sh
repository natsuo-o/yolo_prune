#!bin/bash

parent_dir=$(dirname "$(pwd)")
docker run -it -d \
    --name yolo_container \
    -p 2251:22 \
    --runtime=nvidia \
    --shm-size=256gb \
    -v $parent_dir:/workspace \
    -v ~/.bashrc:/root/.bashrc \
    okamoto/yolo_image 

    # -v /mnt/nfsshare1/datasets-share/nerf/kandao_movie/KD_20240228_193649_MP4:/workspace/NeRF-tutorial/datasets \