import os
import sys
import argparse

import math

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from hw3 import labels_only


def parameterize_file_name(args, file_name):
    return "{}_words{}_strategy{}_l2{}_loss{}".format(file_name, args.num_words,
                                                      args.strategy, args.l2_weight,
                                                      args.loss_improvement)


def command(words, strategy, l2_weight, loss_improvement,
            original_model_fname, minimal_frequency, is_demo):
    FORMAT = "nohup python hw3/labels_only.py --original_model_file_name={0} --stolen_model_file_name=stolen_words{" \
             "8}_method{7}_l2_weight{2}_impr{3} --eta={1} " \
             "--l2_weight={2} --loss_improvement={3} --minimal_frequency={4} --total_queries_amount={5} " \
             "--batch_size={6} " \
             "--strategy={7} --first_random={9} --num_words={8} >& ~/outputs/output_labels_only_freq{4}_words{" \
             "8}_l2_weight{2}_lossimpr{3}_eta{1}_{7}.txt & "
    if is_demo:
        max_queries = 7000
        batch_size = 700
        first_random = 6000
    else:
        max_queries = 150000
        batch_size = 2500
        first_random = 10000
    return FORMAT.format(original_model_fname, 4., l2_weight, loss_improvement, minimal_frequency, max_queries,
                         batch_size, strategy, words, first_random)


def main():
    parser = argparse.ArgumentParser("Creates command run file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs_per_file", required=True, type=int)
    parser.add_argument("--freq", required=True, type=int)
    parser.add_argument("--is_demo", type=bool)
    args = parser.parse_args(sys.argv[1:])
    commands = []

    for words in [2000, 3000]:
        for strategy in labels_only.STRATEGIES:
            for l2_weight in [0.0, 1e-7]:
                for loss_improvement in [1e-2, ]:
                    command_line = command(words, strategy, l2_weight, loss_improvement,
                                           "all_freq{}_my.pkl".format(args.freq), args.freq, args.is_demo)
                    commands.append(command_line)

    n_jobs = args.jobs_per_file
    for i in xrange(int(math.ceil(len(commands) / float(args.jobs_per_file)))):
        fname = args.output + str(i) + ".sh"
        with open(fname, "wb") as f:
            f.write("#! /bin/bash\n\n")
            f.write("cd steal-nlp\n")
            f.write("\n".join(commands[n_jobs * i: n_jobs * (i + 1)]))
            f.write("\n")
            f.write("\n\ndisown -h\n")
            f.write("\n\necho DONE\n")
        os.chmod(fname, 0764)


if __name__ == '__main__':
    main()
