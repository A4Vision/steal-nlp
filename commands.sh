#! /bin/bash

python hw3/full_information_experiments.py --classifier_file_name all_freq30.pkl --minimal_frequency 30 --experiment_number 1 --stolen_fname stolen_exp1_1hour --maximal_queries 1000000 --first_batch_size 10000 --batch_size 5000 --search_minutes 60 >& ~/outputs/output_stolen_freq30_exp1_1hour.txt &
python hw3/full_information_experiments.py --classifier_file_name all_freq20.pkl --minimal_frequency 20 --experiment_number 1 --stolen_fname stolen_exp1_1hour --maximal_queries 1000000 --first_batch_size 10000 --batch_size 5000 --search_minutes 60 >& ~/outputs/output_stolen_freq20_exp1_1hour.txt &
python hw3/full_information_experiments.py --classifier_file_name all_freq10.pkl --minimal_frequency 10 --experiment_number 1 --stolen_fname stolen_exp1_1hour --maximal_queries 1000000 --first_batch_size 10000 --batch_size 5000 --search_minutes 60 >& ~/outputs/output_stolen_freq10_exp1_1hour.txt &

python hw3/full_information_experiments.py --classifier_file_name all_freq30.pkl --minimal_frequency 30 --experiment_number 2 --stolen_fname stolen_exp2_1hour --maximal_queries 100000 --first_batch_size 1000 --batch_size 1000 --search_minutes 60 >& ~/outputs/output_stolen_freq30_exp2_1hour.txt &
python hw3/full_information_experiments.py --classifier_file_name all_freq20.pkl --minimal_frequency 20 --experiment_number 2 --stolen_fname stolen_exp2_1hour --maximal_queries 100000 --first_batch_size 1000 --batch_size 1000 --search_minutes 60 >& ~/outputs/output_stolen_freq20_exp2_1hour.txt &
python hw3/full_information_experiments.py --classifier_file_name all_freq10.pkl --minimal_frequency 10 --experiment_number 2 --stolen_fname stolen_exp2_1hour --maximal_queries 100000 --first_batch_size 1000 --batch_size 1000 --search_minutes 60 >& ~/outputs/output_stolen_freq10_exp2_1hour.txt &


python hw3/full_information_experiments.py --classifier_file_name all_freq30.pkl --minimal_frequency 30 --experiment_number 3 --stolen_fname stolen_exp3_1hour --maximal_queries 100000 --first_batch_size 1000 --batch_size 1000 --search_minutes 60 >& ~/outputs/output_stolen_freq30_exp3_1hour.txt &
python hw3/full_information_experiments.py --classifier_file_name all_freq20.pkl --minimal_frequency 20 --experiment_number 3 --stolen_fname stolen_exp3_1hour --maximal_queries 100000 --first_batch_size 1000 --batch_size 1000 --search_minutes 60 >& ~/outputs/output_stolen_freq20_exp3_1hour.txt &
python hw3/full_information_experiments.py --classifier_file_name all_freq10.pkl --minimal_frequency 10 --experiment_number 3 --stolen_fname stolen_exp3_1hour --maximal_queries 100000 --first_batch_size 1000 --batch_size 1000 --search_minutes 60 >& ~/outputs/output_stolen_freq10_exp3_1hour.txt &
