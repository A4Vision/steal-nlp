import collections
import fractions
import glob
import os
import sys

import re
from matplotlib import pyplot as plt


def last_occurence(li, a):
    return len(li) - 1 - li[::-1].index(a)


class OutputParser(object):
    def __init__(self, content):
        assert isinstance(content, str)
        self._content = content
        self._lines = self._content.splitlines()

    def is_done(self):
        return 'Labels Only Experiment Done' in self._content

    def data_of_row(self, row_header):
        if row_header in self._lines:
            i = last_occurence(self._lines, row_header)
            return eval(self._lines[i + 1])


def rows_data(files_names, rows_headers):
    res = collections.defaultdict(lambda: collections.defaultdict(None))
    for fname in files_names:
        with open(fname, "rb") as f:
            parser = OutputParser(f.read())
            if parser.is_done():
                for row_header in rows_headers:
                    values = parser.data_of_row(row_header)
                    if values is not None:
                        res[row_header][fname] = values
    return res


title_pattern = re.compile(".*words(\\d+)_l2_weight([^_]+).*lossimpr([^_]+).*0_(.+)\\.txt")


def simplify_title(title):
    l = title_pattern.findall(title)
    assert len(l) == 1
    words, reg_coef, improvement, method = l[0]
    return "{},impr={},reg={}".format(method, improvement, reg_coef)


def plot_data(xs, ys_lists, titles, xlabel, ylabel, colors, markers):
    my_dpi = 96
    fig = plt.figure(figsize=(1200 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    ax = plt.subplot(111)

    for title, marker, color, y, x in zip(titles, markers, colors, ys_lists, xs):
        print title
        print len(x), len(y)
        print x
        print y
        x = x[5:]
        y = y[5:]
        ax.plot(x, y, '--', marker=marker, label=title, color=color)
    ax.set_xlabel(xlabel)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    ax.set_ylabel(ylabel)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), prop={'size': 11})


def main():
    patterns = sys.argv[1:]
    files = []
    for pattern in patterns:
        assert pattern.endswith("txt")
        files += glob.glob(pattern)
        print pattern
    print files
    dirname = os.path.dirname(files[0])
    assert all(os.path.dirname(f) == dirname for f in files)

    x_name = "Single word queries amounts"
    ys_names = ("unique_words_amounts",
                "accuracies", "l2 distances", "validation KL", "w l2 norm",
                "norm percent from loss")
    colors = ["red", "green", "blue", "orange", "black", "purple", "yellow", "gray", "pink"]
    markers = ['+', 'o', 's', '.']
    assert fractions.gcd(len(markers), len(colors)) == 1
    colors *= len(markers)
    markers *= len(colors)
    assert len(colors) >= len(ys_names)
    data = rows_data(files, ys_names + (x_name,))
    valid_filenames = sorted(set(sum([v.keys() for v in data.values()], [])))
    print valid_filenames
    for plot_type in ys_names:
        plt.clf()
        plt.cla()
        titles, ys_lists = zip(*sorted(data.get(plot_type, {}).items()))
        xs = [data.get(x_name, {})[title] for title in titles]
        # TODO(bugabuga): read all the parameters from the output file.
        titles_simple = map(simplify_title, titles)
        print 'plot_type', plot_type
        plot_data(xs, ys_lists, titles_simple, "#queries", plot_type, colors, markers)

        plt.savefig(dirname + "/" + plot_type + ".png")


if __name__ == '__main__':
    main()

