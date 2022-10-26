#!/usr/bin/env bash
set -x

HADOOP_STREAMING_JAR=/opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar
HDFS_OUTPUT_DIR_MEAN=/user/out/mean
HDFS_OUTPUT_DIR_VAR=/user/out/var

hdfs dfs -rm -r -skipTrash $HDFS_OUTPUT_DIR_MEAN

yarn jar $HADOOP_STREAMING_JAR \
        -D mapreduce.job.name="HW1" \
        -D mapreduce.job.maps=4 \
        -files mapper_mean.py,reducer_mean.py \
        -mapper 'python3 mapper_mean.py' \
        -reducer 'python3 reducer_mean.py' \
        -input AB_NYC_2019.csv \
        -output $HDFS_OUTPUT_DIR_MEAN

hdfs dfs -rm -r -skipTrash $HDFS_OUTPUT_DIR_VAR

yarn jar $HADOOP_STREAMING_JAR \
        -D mapreduce.job.name="HW2" \
        -D mapreduce.job.maps=4 \
        -files mapper_var.py,reducer_var.py \
        -mapper 'python3 mapper_var.py' \
        -reducer 'python3 reducer_var.py' \
        -input AB_NYC_2019.csv \
        -output $HDFS_OUTPUT_DIR_VAR