IMAGE_NAME := blackjack-ai
DOCKERFILE_PATH = .
WORKDIR := /workspace

all: build run

build:
	docker build -t $(IMAGE_NAME) $(DOCKERFILE_PATH)

run:
	docker run -it --gpus all -v $(shell pwd):$(WORKDIR) $(IMAGE_NAME) /bin/bash

stop:
	docker stop $(CONTAINER_NAME)

clean:
	docker rm $(CONTAINER_NAME)

clean-image:
	docker rmi $(IMAGE_NAME)

.PHONY: all build run stop clean clean-image
