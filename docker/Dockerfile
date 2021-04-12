FROM quay.io/fenicsproject/dev:latest
MAINTAINER Nate Sime <njcs4@cam.ac.uk>

RUN git clone \
    https://bitbucket.org/nate-sime/leopart.git --depth 1 && \
    cd leopart/source/cpp && cmake . && make && cd ../.. && \
    sudo python3 setup.py install