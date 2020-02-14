name := "glint-fmpair"

version := "1.0"

organization := "com.github.mgabr"

scalaVersion := "2.11.8"
val scalaMajorMinorVersion = "2.11"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0" % "provided"

// use this instead of the github dependency for easier local development if you are modifying glint
// libraryDependencies += "com.github.mgabr" %% "glint" % "0.2-SNAPSHOT"

lazy val glint = RootProject(uri("https://github.com/MGabr/glint.git#0.2-fmpair"))

// Integration tests

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.8" % "it"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.8" % "it"

// Add it:assembly task to build separate jar containing only the integration test sources

Project.inConfig(IntegrationTest)(baseAssemblySettings)
assemblyJarName in (IntegrationTest, assembly) := s"${name.value}-it-assembly-${version.value}.jar"
test in (IntegrationTest, assembly) := {}
fullClasspath in (IntegrationTest, assembly) := {
  val cp = (fullClasspath in (IntegrationTest, assembly)).value
  cp.filter({ x => Seq("it-classes", "scalatest", "scalactic").exists(x.data.getPath.contains(_)) })
}

// Override it:test task to execute integration tests in Spark docker container

val sparkParameterServerArg = s"spark -c target/scala-$scalaMajorMinorVersion/it-classes/separate-glint.conf"
val sparkParameterServerMain = "glint.Main"
val sparkTestsMain = "org.apache.spark.ml.recommendation.Main"

import scala.sys.process._

test in IntegrationTest := {
  val startSparkTestEnv = "./spark-test-env.sh"
  val execSparkParameterServer =
    s"""./spark-test-env.sh exec-detach
        spark-submit
        --conf spark.driver.extraJavaOptions=-XX:+UseG1GC
        --driver-memory 512m
        --conf spark.executor.extraJavaOptions=-XX:+UseG1GC
        --executor-memory 512m
        --total-executor-cores 2
        --class $sparkParameterServerMain
        target/scala-$scalaMajorMinorVersion/${name.value}-assembly-${version.value}.jar
        $sparkParameterServerArg
     """
  val execSparkTests =
    s"""./spark-test-env.sh exec
        spark-submit
        --driver-memory 1024m
        --executor-memory 1024m
        --total-executor-cores 2
        --jars target/scala-$scalaMajorMinorVersion/${name.value}-assembly-${version.value}.jar
        --class $sparkTestsMain
        target/scala-$scalaMajorMinorVersion/${name.value}-it-assembly-${version.value}.jar
    """
  val stopSparkTestEnv = "./spark-test-env.sh stop"
  val rmSparkTestEnv = "./spark-test-env.sh rm"
  val exitCode = (startSparkTestEnv #&& execSparkParameterServer #&& execSparkTests #&& stopSparkTestEnv #&& rmSparkTestEnv !)
  if (exitCode != 0) {
    (stopSparkTestEnv ### rmSparkTestEnv !)
    throw new RuntimeException(s"Integration tests failed with nonzero exit value: $exitCode")
  }
}

test in IntegrationTest := (test in IntegrationTest).dependsOn(
  assembly,
  assembly in IntegrationTest
).value

// Add integration tests to sbt project

lazy val root = (project in file("."))
  .dependsOn(glint)
  .configs(IntegrationTest)
  .settings(Defaults.itSettings)


// Scala documentation

scalacOptions in(Compile, doc) ++= Seq("-doc-title", "Glint-FMPair")

enablePlugins(GhpagesPlugin)

git.remoteRepo := "git@github.com:MGabr/glint-fmpair.git"

enablePlugins(SiteScaladocPlugin)

