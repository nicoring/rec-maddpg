#!/bin/sh
#$ -N test
#$ -cwd
#$ -l h_rt=24:00:00

# Initialise the environment modules
. /etc/profile.d/modules.sh
 
# Load Anaconda Python
module load anaconda/5.0.1

source activate thesis
 
"$@"