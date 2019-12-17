# Glint-FMPair

> Network-efficient distributed pairwise factorization machines for large models on Spark
using customized [Glint](https://github.com/MGabr/glint) parameter servers

This Spark ML estimator is a distributed implementation of pairwise factorization machines,
specifically the model presented as LightFM. Important features include

* **Distributed large models:**
Training is done as asynchronous mini-batch gradient descent using parameter servers.
The model is distributed to the parameter servers
and therefore does not need to fit on a single machine.

* **Network-efficient training:**
This implementation employs tricks similar to those from distributed Word2Vec training to enable network-efficient training.
The latent factor dimensions and the active features are aggregated on the parameter servers,
so it is not necessary to transmit all latent factor dimensions of all active features.

* **Different sampling methods:**
Different methods of sampling negative items can be used.
Besides the common BPR loss with negative items sampled uniformly,
negative items can also be sampled with probability proportional to their popularity
and different methods for accepting negative items for a user are supported.
For example, for playlist continuation, one method can be to only accept tracks whose artist is not in the playlist.

Large parts of the actual functionality can be found in the [Glint fork](https://github.com/MGabr/glint).

## Build

You can either use the [release jar](https://github.com/MGabr/glint-fmpair/releases) or build the project yourself.
To build it run:

    sbt assembly
 
To also execute integration tests run:

    sbt it:test

The resulting fat jar contains all dependencies and can be passed to `spark-submit` with `--jars`.

## Usage

The API is similar to the existing ALS implementation of Spark
with specific parameters for FMPair. The API implements the Spark ML interfaces.

There are two modes in which Glint-FMPair can be run.
One can either start the parameter servers automatically on some executors in the same Spark application
or start up a parameter server cluster in a separate Spark application beforehand
and then specify the IP of the parameter server master to use this cluster.
The first mode is more convenient but the second mode scales better
so it is recommended to use a separate parameter server for training.
The integrated parameter servers can be used for recommendation.

To start parameter servers as separate Spark application run:

    spark-submit --num-executors num-servers --executor-cores server-cores --class glint.Main /path/to/compiled/Glint-FMPair.jar spark

The parameter server master will be started on the driver and the drivers IP will be written to the log output.
Pass this IP as `parameterServerHost` to connect to these parameter servers from the Glint-FMPair Spark application. 

## Spark parameters

Use a higher number of executor cores (`--executor-cores`) instead of more executors (`--num-executors`).
Preferably set `--executor-cores` to the number of available virtual cores per machine.

Each of the n parameter servers will require enough executor memory (`--executor-memory`) to store 1/n of the latent factors matrix.

Further, enough memory for storing the mapping array of item indices to item features is required on each executor.
It is used for sampling negative items and has to be broadcasted and therefore be below the 8GB broadcast size limit of Spark.