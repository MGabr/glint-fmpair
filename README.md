# Glint-FMPair

> Network-efficient distributed pairwise factorization machines for large models on Spark
using customized [Glint](https://github.com/MGabr/glint) parameter servers

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

