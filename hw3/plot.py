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


title_pattern = re.compile(".*words(\\d+).*lossimpr([^_]+).*0_(.+)\\.txt")


def simplify_title(title):
    l = title_pattern.findall(title)
    assert len(l) == 1
    words, improvement, method = l[0]
    return "method={},Words={},impr={}".format(method, words, improvement)


def plot_data(xs, ys_lists, titles, xlabel, ylabel, colors):
    fig = plt.figure()
    ax = plt.subplot(111)

    for title, color, y, x in zip(titles, colors, ys_lists, xs):
        print len(x), len(y)
        print x
        print y
        ax.plot(x, y, label=title, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.3, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def main():
    dirname = sys.argv[1]
    assert os.path.isdir(dirname)
    files = glob.glob(dirname + "/*.txt")
    x_name = "Single word queries amounts"
    ys_names = ("unique_words_amounts",
                "accuracies", "l2 distances", "validation KL")
    colors = ["red", "green", "blue", "orange", "black", "purple"] * 4
    assert len(colors) >= len(ys_names)
    data = rows_data(files, ys_names + (x_name,))
    valid_filenames = sorted(set(sum([v.keys() for v in data.values()], [])))
    title2color = dict(zip(valid_filenames, colors))
    for plot_type, color in zip(ys_names, colors):
        plt.clf()
        plt.cla()
        titles, ys_lists = zip(*data.get(plot_type, {}).iteritems())
        xs = [data.get(x_name, {})[title] for title in titles]
        titles_simple = map(simplify_title, titles)
        plot_data(xs, ys_lists, titles_simple, "#queries", plot_type, [title2color[t] for t in titles])
        plt.savefig(dirname + "/" + plot_type + ".png")



if __name__ == '__main__':
    main()