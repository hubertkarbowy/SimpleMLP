#!/bin/bash

# This script generates plots in out/plots dirrectory and csv logs in out directory.

PRJDIR="$(pwd)/../../.."
JDK="/usr/java/jdk-13.0.1/bin"
CLASSPATH="${PRJDIR}/out/production/SimpleMLP:${HOME}/.m2/repository/com/beust/jcommander/1.78/jcommander-1.78.jar"
TRAINSET="${HOME}/Downloads/mnistasjpg/trainingSet/trainingSet/"
OUTDIR="${PRJDIR}/out"
TRAINOUT="${OUTDIR}/trained"
PLOTDIR="${OUTDIR}/plots"

ITER_FILE="${OUTDIR}/iter.csv"
DELTA_FILE="${OUTDIR}/delta.csv"
PERC_FILE="${OUTDIR}/perc.csv"
PATIENCE_FILE="${OUTDIR}/patience.csv"
LAYERS_FILE="${OUTDIR}/layers.csv"
N_EXAMPLES_FILE="${OUTDIR}/examples.csv"

RUNCOMMAND="${JDK}/java -Dfile.encoding=UTF-8 -classpath ${CLASSPATH} pl.hubertkarbowy.Main"

mkdir ${TRAINOUT}
mkdir ${PLOTDIR}

# Prepare
${RUNCOMMAND} -trainset ${TRAINSET} -truelabel "3" -outdir ${TRAINOUT} -split

# Train
TRAIN="${RUNCOMMAND} -trainset "${TRAINOUT}/train/" -outdir "${OUTDIR}/net.ser" -train"

# Run
RUN="${RUNCOMMAND} -testset "${TRAINOUT}/test/" -pretrained "${OUTDIR}/net.ser" -runeval"

run_tests () {
  XS=$1
  PARAM=$2
  FILE=$3
  for i in ${XS}
  do
    ${TRAIN} ${PARAM} $i
    F1=$(${RUN} | grep 'F1' -A3 | cut -d ':' -f 2 | tr -d ' ')
    echo "$i ${F1}" >> ${FILE}
  done

  # Make plot
  gnuplot <<EOL
  set terminal png
  set output "${PLOTDIR}/${PARAM}.png"
  set xlabel "Iterations"
  set ylabel "F1"
  set logscale x 10
  plot "${FILE}" title with line "Net F1"
EOL
}

ITERATIONS="10 100 1000 10000"
DELTAS="0.01 0.05 0.1 0.5 1"
PERCS="0.01 0.1 1 10"
PATIENCES="10 100 1000"
LAYERS="1 10 100"
TRAINS="10 100 1000"

run_tests "${ITERATIONS}" "-max-iter" "${ITER_FILE}"
run_tests "${DELTAS}" "-max-delta" "${DELTA_FILE}"
run_tests "${PERCS}" "-perc-change" "${PERC_FILE}"
run_tests "${PATIENCES}" "-max-patience" "${PATIENCE_FILE}"
run_tests "${ITERATIONS}" "-layers" "${LAYERS_FILE}"

for i in ${TRAINS}
do
  rm -rf ${TRAINOUT}
  ${RUNCOMMAND} -trainset ${TRAINSET} -truelabel "3" -outdir ${TRAINOUT} -split -trains $i
  run_tests "$i" "-trains" "${LAYERS_FILE}"
done