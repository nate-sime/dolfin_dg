FROM quay.io/fenicsproject/dev:latest
MAINTAINER Nate Sime <njcs4@cam.ac.uk>

USER fenics

RUN bash -c "source ~/fenics.env.conf &&\
            git clone https://bitbucket.org/nate-sime/dolfin_dg.git &&\
            cd dolfin_dg &&\
            sudo python3 setup.py install"

USER root