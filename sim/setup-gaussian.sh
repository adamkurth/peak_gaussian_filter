#!/bin/bash

NAME=$1
STEP=7

mkdir ${NAME}
cd ${NAME}
ln -s ../Eiger.geom .
ln -s ../PYP.cell .
j=1

for i in $(seq 1 $STEP 50); do
    mkdir step${j}
    cd step${j}
    START=$i
    END=$((102-$START))
    ln -s ../../pattern_sim_submit .                                        
    for k in $(seq $START 1 $END); do
        ln -s  ../../frame${k}.hkl ${NAME}-${j}00${k}.hkl 
        ./pattern_sim_submit ${NAME}-${j}00${k} 40 htc normal
    done
    cd ..
    j=$((j+1))
done

cd ..

