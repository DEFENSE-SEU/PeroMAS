#!/bin/bash

cd "/seu_share2/home/dishimin/230259983/psc_agent/Perovskite_PI_Multi"

source "/seu_share2/home/dishimin/230259983/Anaconda/etc/profile.d/conda.sh"
# conda activate /seu_share2/home/dishimin/230259983/Anaconda/envs/psc-pi2
conda activate /seu_share2/home/dishimin/230259983/Anaconda/envs/psc-pi-fastmcp
# bash /seu_share2/home/dishimin/230259983/psc_agent/Perovskite_PI_Multi/run.sh


python main1_hyperopt.py

python main2_train.py

python main3_predict.py

python main3_interpret.py