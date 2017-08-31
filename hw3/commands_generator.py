import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from hw3 import labels_only


def parameterize_file_name(args, file_name):
    return "{}_words{}_strategy{}_l2{}_loss{}".format(file_name, args.num_words,
                                                      args.strategy, args.l2_weight,
                                                      args.loss_improvement)


def command(words, strategy, l2_weight, loss_improvement,
            original_model_fname, minimal_frequency):
    FORMAT = "python hw3/labels_only.py --original_model_file_name={0} --stolen_model_file_name=stolen --eta={1} " \
             "--l2_weight={2} --loss_improvement={3} --minimal_frequency={4} --total_queries_amount={5} " \
             "--batch_size={6} " \
             "--strategy={7} --first_random=1000 --num_words={8} >& ~/outputs/output_labels_only_freq{4}_words{" \
             "8}_l2_weight{2}_lossimpr{3}_eta{1}_{7}.txt & "
    return FORMAT.format(original_model_fname, 6., l2_weight, loss_improvement, minimal_frequency, 100000,
                         600, strategy, words)


def main():
    parser = argparse.ArgumentParser("Creates command run file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[1:])
    commands = []

    for words in range(1000, 3001, 500):
        for strategy in labels_only.STRATEGIES:
            for l2_weight in [0.01, 0.0001]:
                for loss_improvement in [1e-4, 1e-6]:
                    command_line = command(words, strategy, l2_weight, loss_improvement, "all_freq20.pkl", 20)
                    commands.append(command_line)
    for i in xrange(len(commands) / 3):
        with open(args.output + str(i) + ".sh", "wb") as f:
            f.write("#! /bin/bash\n\n")
            f.write("\n".join(commands[3 * i: 3 * (i + 1)]))
            f.write("\n")
            f.write("\n\ndisown -h\n")


if __name__ == '__main__':
    main()
