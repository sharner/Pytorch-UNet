#!/usr/bin/env bash

docker run \
	--gpus all \
	--mount type=bind,source="$(pwd)",target=/layerjot/Pytorch-UNet \
	--mount type=bind,source="/data/caravana",target=/data \
	--rm --ipc=host -it pytorch-unet:latest
