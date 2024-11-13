#!/bin/bash
echo "Running the script"
counter=1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
echo "Experiment: $counter"
python scripts/train_lstm.py
((counter++))
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
echo "Experiment: $counter"
python scripts/train_vlstm.py
((counter++))