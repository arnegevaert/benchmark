#!/bin/bash

source /home/arne/miniconda3/etc/profile.d/conda.sh
conda activate benchmark

python sample_selection.py
python attributions_computation.py

python distributed_deletion.py

python distributed_impact_coverage.py

python distributed_infidelity.py

python distributed_irof.py

python distributed_max_sens.py

python distributed_minimal_subset.py

python distributed_seg_sens_n.py

python distributed_sens_n.py
