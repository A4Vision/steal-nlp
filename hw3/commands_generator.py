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
            original_model_fname, minimal_frequency, total_runtime, is_demo, no_forward):
    COMMAND = "python hw3/labels_only.py --original_model_file_name={0} --stolen_model_file_name=stolen_words{" \
              "8}_method{7}_l2_weight{2}_impr{3} --eta={1} " \
              "--l2_weight={2} --loss_improvement={3} --minimal_frequency={4} --total_queries_amount={5} " \
              "--batch_size={6} --max_batch_time_secs={10:.0f} " \
              "--strategy={7} --first_random={9} --num_words={8} "
    FORWARD_OUTPUT = ">& ~/outputs/output_labels_only_freq{4}_words{" \
                     "8}_l2_weight{2}_lossimpr{3}_eta{1}_{7}.txt & "
    if is_demo:
        max_queries = 7000
        batch_size = 700
        first_random = 6000
    else:
        max_queries = 200000
        batch_size = 5000
        first_random = 10000
    if no_forward:
        full_command = COMMAND
    else:
        full_command = "nohup " + COMMAND + FORWARD_OUTPUT
    batches_amount = (max_queries - first_random) // batch_size
    max_batch_time = total_runtime / batches_amount
    return full_command.format(original_model_fname, 4., l2_weight, loss_improvement, minimal_frequency, max_queries,
                               batch_size, strategy, words, first_random, max_batch_time)


def main():
    parser = argparse.ArgumentParser("Creates command run file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs_per_file", required=True, type=int)
    parser.add_argument("--freq", required=True, type=int)
    parser.add_argument("--total_runtime", required=True, type=float)
    parser.add_argument("--is_demo", type=bool)
    parser.add_argument("--no_forward", type=bool)
    args = parser.parse_args(sys.argv[1:])
    commands = []

    for words in [2000, 3000]:
        for l2_weight in [0.0, 1e-7]:
            # Require 2% improvement in the training loss.
            for loss_improvement in [0.02]:
                # The strategies at the end, to put different strategies on the same machine
                for strategy in labels_only.STRATEGIES:
                    command_line = command(words, strategy, l2_weight, loss_improvement,
                                           "all_freq{}_my.pkl".format(args.freq), args.freq, args.total_runtime, args.is_demo,
                                           args.no_forward)
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
