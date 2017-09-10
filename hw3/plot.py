import collections
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
        return 'W shape:' in self._content

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
    return "{},#W={},impr={},reg={}".format(method, words, improvement, reg_coef)


def plot_data(xs, ys_lists, titles, xlabel, ylabel, colors):
    my_dpi = 96
    fig = plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    ax = plt.subplot(111)

    for title, color, y, x in zip(titles, colors, ys_lists, xs):
        print len(x), len(y)
        print x
        print y
        ax.plot(x, y, '--+', label=title, color=color)
    ax.set_xlabel(xlabel)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    ax.set_ylabel(ylabel)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})


def main():
    pattern = sys.argv[1]
    assert pattern.endswith("txt")
    files = glob.glob(pattern)
    print pattern
    print files
    dirname = os.path.dirname(files[0])
    assert all(os.path.dirname(f) == dirname for f in files)

    x_name = "Single word queries amounts"
    ys_names = ("unique_words_amounts",
                "accuracies", "l2 distances", "validation KL")
    colors = ["red", "green", "blue", "orange", "black", "purple", "magenta", "cyan", "yellow", "gray"] * 4
    assert len(colors) >= len(ys_names)
    data = rows_data(files, ys_names + (x_name,))
    valid_filenames = sorted(set(sum([v.keys() for v in data.values()], [])))
    print valid_filenames
    title2color = dict(zip(valid_filenames, colors))
    for plot_type, color in zip(ys_names, colors):
        plt.clf()
        plt.cla()
        titles, ys_lists = zip(*data.get(plot_type, {}).iteritems())
        xs = [data.get(x_name, {})[title] for title in titles]
        titles_simple = map(simplify_title, titles)
        plot_data(xs, ys_lists, titles_simple, "#queries", plot_type, [title2color[t] for t in titles
                                                                       if t in valid_filenames])

        plt.savefig(dirname + "/" + plot_type + ".png")


if __name__ == '__main__':
    main()
