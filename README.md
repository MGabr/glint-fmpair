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

You can either use the [release jar](https://github.com/MGabr/glint-word2vec/releases) or build the project yourself.
To build it run:

    sbt assembly
 
To also execute integration tests run:

    sbt it:test

The resulting fat jar contains all dependencies and can be passed to `spark-submit` with `--jar`.
To use the python bindings zip them and pass them as `--py-files`.

    cd src/main/python
    zip ml_glintfmpair.zip ml_glintfmpair.py

