FROM ros:noetic-robot

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    git curl wget \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    ultralytics \
    tflite-runtime \
    numpy \
    matplotlib \
    pandas \
    opencv-python-headless

RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws
RUN /bin/bash -c \
    "source /opt/ros/noetic/setup.bash && catkin_make"

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
