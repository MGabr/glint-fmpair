# Glint-FMPair
[![Build Status](https://travis-ci.com/MGabr/glint-fmpair.svg)](https://travis-ci.com/MGabr/glint-fmpair)

> Network-efficient distributed pairwise factorization machines for large models.\
> Spark implementation using customized [Glint](https://github.com/MGabr/glint) parameter servers

This Spark ML estimator is a distributed implementation of pairwise factorization machines
for ranking with implicit feedback, specifically the model presented as [LightFM](https://github.com/lyst/lightfm). 
Important features include

* **Distributed large models:**
Training is done as asynchronous mini-batch gradient descent using parameter servers.
The model is distributed to the parameter servers and therefore does not need to fit on a single machine.

* **Network-efficient training:**
This implementation employs tricks similar to those from distributed Word2Vec training to enable network-efficient 
training. The latent factor dimensions and the active features are aggregated on the parameter servers,
so it is not necessary to transmit all latent factor dimensions of all active features.

* **Different negative feedback sampling methods:**
Different methods of sampling negative items can be used.
Besides the common BPR loss with negative items sampled uniformly,
negative items can also be sampled with probability proportional to their popularity
and different methods for accepting negative items for a user are supported.
For example, for playlist continuation, one method can be to only accept items whose artist is not in the playlist.

* **Mini-batch sharing of negative feedback:** 
Another alternative sampling method which can be used is crossbatch-BPR.
This method samples negative feedback uniformly but then shares the negative feedback across the mini-batch.
This can lead to significantly faster training convergence (e.g. x10 in some evaluations)
but also leads to less network-efficient training.

Large parts of the actual functionality can be found in the 
[Glint fork](https://github.com/MGabr/glint/tree/0.2-fmpair).

This implementation was used in an evaluation for 
[next-track music recommendation](https://github.com/MGabr/fm-next-songs-rec).

## Build

You can either use the [release jar](https://github.com/MGabr/glint-fmpair/releases) or build the project yourself.
To build it run:

    sbt assembly
 
To also execute integration tests run:

    sbt it:test

The resulting fat jar contains all dependencies and can be passed to `spark-submit` with `--jars`.

## Usage

The API implements the Spark ML interfaces and is similar to the API of the Spark ML ALS recommender.

To use `GlintFMPair` in a Spark application, 
the parameter servers have to be started beforehand in a separate Spark application.
To start 4 parameter servers with 40GB memory and 40 cores each on YARN, run:

    spark-submit \
        --master yarn --deploy-mode cluster --driver-memory 10G \
        --num-executors 4 --executor-cores 40 --executor-memory 40G \
        --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC" \
        --class glint.Main glint-fmpair-assembly-1.0.jar spark

The parameter server master will be started on the driver and the master IP will be written to the log output.
Pass this IP as `parameterServerHost` to connect to the parameter servers 
when training with `GlintFMPair` in another Spark application.

The trained `GlintFMPairModel` can then recommend top items for users. 

This package also contains some Spark ML baseline recommenders (`PopRank`, `SAGH` and `KNN`)
as well as an `WeightHotEncoderEstimator` to one-hot encode features with weights other than one
and to encode groups of features in a weighted way.

More information can be found in the 
[Scaladoc](https://mgabr.github.io/glint-fmpair/latest/api)
of this project.

Scala examples can be found in the 
[integration tests](https://github.com/MGabr/glint-fmpair/blob/master/src/it/scala/org/apache/spark/ml/recommendation/GlintFMPairSpec.scala) 
and Python examples can be found in an 
[next-track music recommendation](https://github.com/MGabr/fm-next-songs-rec) evaluation using this implementation.

## Things to note

The implementation is only a prototype.
There are some things to note and a lot of room for improvement.

* **Executor memory - mappings of GlintFMPair / workers:**
For the sampling of negative feedback, a mapping of item ids to item features is created.
This mapping is broadcasted to all workers and therefore has to be below the 8GB Spark broadcast limit. 
For large mappings, it might be better to sample negative feedback only from the items on a worker 
or share the positive feedback of a mini-batch as negative feedback.
Such an approach would still have to be implemented and evaluated though.

* **Executor memory and executor cores - parameter servers:** 
The reduction in network traffic is proportional to the ratio 
between the number of latent factors and the number of parameter servers.
So it is recommended to use only few executors with a high memory and many cores for the parameter servers.

* **Executor memory and executor cores - GlintFMPair / workers:** 
Using Spark with more than 5 executor cores can lead to HDFS throughput problems
significantly slowing down the Spark application. 
Using multiple executors on a machine means that each executor needs its own copy of the mappings used for sampling.
This leads to higher memory requirements per machine. A good balance has to be found.
For the separate parameter server application this is not an issue.

* **Efficiency of recommendations:** 
For simplicity, recommendation are also made using the parameter servers.
This requires parameter servers running at recommendation time and takes linear time.
For real use cases, it would be better to use localitiy-senstive hashing to make recommendations in sublinear time
and not rely on parameter servers at recommendation time.

* **Efficiency of implementation:** 
Apart from general code efficiency there could be better avoidance of skew when distributing data to workers,
avoidance of garbage collection on servers, good compression and gradient quantization
as well as the eventual use of the Aeron transport protocol.
