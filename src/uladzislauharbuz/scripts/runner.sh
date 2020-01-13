#!/bin/bash

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

RUNCOMMAND="${JDK}/java -Dfile.encoding=UTF-8 -classpath ${CLASSPATH} pl.hubertkarbowy.Main"

mkdir ${TRAINOUT}
mkdir ${PLOTDIR}

# Prepare
${RUNCOMMAND} -trainset ${TRAINSET} -truelabel "3" -outdir ${TRAINOUT} -split

# Train
TRAIN="${RUNCOMMAND} -trainset "${TRAINOUT}/train/" -outdir "${OUTDIR}/net.ser" -train"

# Run
RUN="${RUNCOMMAND} -testset "${TRAINOUT}/test/" -pretrained "${OUTDIR}/net.ser" -runeval"

ITERATIONS="10 100 1000"

# Get statistics for different iteration number
for i in ${ITERATIONS}
do
  ${TRAIN} -max-delta 0.03 -perc-change 0.05 -max-patience 30 -max-iter $i
  F1=$(${RUN} | grep 'F1' -A3 | cut -d ':' -f 2 | tr -d ' ')
  echo "$i ${F1}" >> ${ITER_FILE}
done

# Make plot
gnuplot <<EOL
set terminal png
set output "${PLOTDIR}/iter.png"
set xlabel "Iterations"
set ylabel "F1"
plot "${ITER_FILE}" title "Net F1"
EOL

# Get statistics for different maxDelta
DELTAS="0.01 0.1 1 10"

for i in ${DELTAS}
do
  ${TRAIN} -max-delta $i -perc-change 0.05 -max-patience 30 -max-iter 5000
  F1=$(${RUN} | grep 'F1' -A3 | cut -d ':' -f 2 | tr -d ' ')
  echo "$i ${F1}" >> ${DELTA_FILE}
done

# Make plot
gnuplot <<EOL
set terminal png
set output "${PLOTDIR}/delta.png"
set xlabel "Maximum delta"
set ylabel "F1"
plot "${DELTA_FILE}" title "Net F1"
EOL

# Get statistics for different percent of changes
PERCS="0.01 0.1 1 10"

for i in ${PERCS}
do
  ${TRAIN} -max-delta 0.03 -perc-change $i -max-patience 30 -max-iter 5000
  F1=$(${RUN} | grep 'F1' -A3 | cut -d ':' -f 2 | tr -d ' ')
  echo "$i ${F1}" >> ${PERC_FILE}
done

# Make plot
gnuplot <<EOL
set terminal png
set output "${PLOTDIR}/perc.png"
set xlabel "Maximum percent of changes"
set ylabel "F1"
plot "${PERC_FILE}" title "Net F1"
EOL

# Get statistics for different patience
PATIENCES="10 100 1000"

for i in ${PATIENCES}
do
  ${TRAIN} -max-delta 0.03 -perc-change 0.05 -max-patience $i -max-iter 5000
  F1=$(${RUN} | grep 'F1' -A3 | cut -d ':' -f 2 | tr -d ' ')
  echo "$i ${F1}" >> ${PATIENCE_FILE}
done

# Make plot
gnuplot <<EOL
set terminal png
set output "${PLOTDIR}/patience.png"
set xlabel "Maximum patience"
set ylabel "F1"
plot "${PATIENCE_FILE}" title "Net F1"
EOL