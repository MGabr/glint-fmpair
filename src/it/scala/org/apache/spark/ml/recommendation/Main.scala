package org.apache.spark.ml.recommendation

import org.scalatest.tools.Runner

/**
 * Entry point for integration tests on Spark
 */
object Main {

  def main(args: Array[String]): Unit = {
    val testResult = Runner.run(Array("-o", "-R", "target/scala-2.11/it-classes"))
    if (!testResult) {
      System.exit(1)
    }
  }
}
