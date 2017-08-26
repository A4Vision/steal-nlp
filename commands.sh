#! /bin/bash

python hw3/full_information_experiments.py --classifier_file_name all_freq30.pkl --minimal_frequency 30 --experiment_number 1 --stolen_fname stolen_exp1_1hour --maximal_queries 1000000 --first_batch_size 10000 --batch_size 10000 --search_minutes 1 >& ~/outputs/output_stolen_freq30 _exp1_1hour.txt
python hw3/full_information_experiments.py --classifier_file_name all_freq20.pkl --minimal_frequency 20 --experiment_number 1 --stolen_fname stolen_exp1_1hour --maximal_queries 1000000 --first_batch_size 10000 --batch_size 10000 --search_minutes 1 >& ~/outputs/output_stolen_freq20 _exp1_1hour.txt
python hw3/full_information_experiments.py --classifier_file_name all_freq10.pkl --minimal_frequency 10 --experiment_number 1 --stolen_fname stolen_exp1_1hour --maximal_queries 1000000 --first_batch_size 10000 --batch_size 10000 --search_minutes 1 >& ~/outputs/output_stolen_freq10 _exp1_1hour.txt

