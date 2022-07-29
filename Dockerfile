#!/bin/bash

FROM amazoncorretto:8

ARG BUILD_DATE
ARG SPARK_VERSION=3.0.0

LABEL org.label-schema.name="Apache PySpark $SPARK_VERSION" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.version=$SPARK_VERSION

ENV PYSPARK_PYTHON="/opt/miniconda3/bin/python"

RUN yum -y update 
RUN yum -y install yum-utils

RUN yum -y list python3*
RUN yum -y install python3 python3-dev python3-pip python3-virtualenv

RUN python -V
RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install pyspark
RUN pip3 install findspark

RUn python3 -c "import numpy as np"

#RUN mkdir /predict
ENV PROG_DIR /winepredict
COPY test.py /winepredict/
COPY ValidationDataset.csv /winepredict/
COPY trainingmodel.model /winepredict/

ENV PROG_NAME test.py
ADD ${PROG_NAME} .

ENTRYPOINT ["spark-submit","test.py"]
CMD ["ValidationDataset.csv"]
