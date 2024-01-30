#!/bin/sh

#SBATCH --time=0-60:00
#SBATCH --ntasks=

#SBATCH --chdir   /Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/peak_gaussian_filter/sim
#SBATCH --job-name  
#SBATCH --output    .out
#SBATCH --error    .err

pattern_sim -g -p --number=10000 -o -i -y 4/mmm -r --min-size=10000 --max-size=10000 --spectrum=tophat -s 7 --background=0 --beam-bandwidth=0.01 --nphotons=3e8 --beam-radius=5e-6
