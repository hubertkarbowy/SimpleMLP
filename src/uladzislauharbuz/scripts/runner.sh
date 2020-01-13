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
TRAIN="${RUNCOMMAND} -trainset "${TRAINOUT}/train/" -outdir "${OUTDIR}/net.ser" -train"

# Run
RUN="${RUNCOMMAND} -testset "${TRAINOUT}/test/" -pretrained "${OUTDIR}/net.ser" -runeval"

${TRAIN} -max-delta 0.03 -perc-change 0.05 -max-patience 30 -max-iter 15000

${RUN}