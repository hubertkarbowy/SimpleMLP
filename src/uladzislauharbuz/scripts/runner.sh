#!/bin/bash

PRJDIR="$(pwd)/../../.."
JDK="/usr/java/jdk-13.0.1/bin"
CLASSPATH="${PRJDIR}/out/production/SimpleMLP:${HOME}/.m2/repository/com/beust/jcommander/1.78/jcommander-1.78.jar"
TRAINSET="${HOME}/Downloads/mnistasjpg/trainingSet/trainingSet/"
OUTDIR="${PRJDIR}/out"
TRAINOUT="${OUTDIR}/trained"

RUNCOMMAND="${JDK}/java -Dfile.encoding=UTF-8 -classpath ${CLASSPATH} pl.hubertkarbowy.Main"

mkdir ${TRAINOUT}

# Prepare
${RUNCOMMAND} -trainset ${TRAINSET} -truelabel "3" -outdir ${TRAINOUT} -split

# Train
${RUNCOMMAND} -trainset "${TRAINOUT}/train/" -outdir "${OUTDIR}/net.ser" -train

# Run
${RUNCOMMAND} -testset "${TRAINOUT}/test/" -pretrained "${OUTDIR}/net.ser" -runeval