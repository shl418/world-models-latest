# USAGE - ./launch-docker-gpu.sh {abs-path-to-WorldModels-code}
docker run --rm -it \
    --gpus all \
    --network=host \
    --volume=$1:/Worldmodels \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  worldmodels-image:latest bash

