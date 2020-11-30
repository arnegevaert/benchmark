#!/bin/bash


python generate_synthetic_data_model.py -f 2 -i 2 -c 3 -n 50 -o ../../data/synthetic/tiny
python generate_synthetic_data_model.py -f 10 -i 8 -c 3 -n 50 -o ../../data/synthetic/small
python generate_synthetic_data_model.py -f 50 -i 40 -c 3 -n 50 -o ../../data/synthetic/medium
python generate_synthetic_data_model.py -f 150 -i 120 -c 3 -n 50 -o ../../data/synthetic/large
