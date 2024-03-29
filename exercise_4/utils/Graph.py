import numpy as np
import matplotlib.pyplot as plt

def draw(filename, x, y, xlim, ylim, title):
    plt.title(title)
    plt.plot(x, y, '-')
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    max_idx = np.argmax(y) + 1
    max_val = round(np.max(y), 2)
    min_idx = np.argmin(y) + 1
    min_val = round(np.min(y), 2)
    plt.plot(max_idx, max_val, marker='^')
    plt.plot(min_idx, min_val, marker='v')
    plt.annotate('({}, {})'.format(max_idx, max_val), (max_idx, max_val))
    plt.annotate('({}, {})'.format(min_idx, min_val), (min_idx, min_val))
    plt.savefig('log/{}.png'.format(filename))
    plt.close()


def drawHyper(filename, x, y, title):
    plt.title(title)
    plt.plot(x, y, '-')
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    max_idx = np.argmax(y) + 1
    max_val = round(np.max(y), 2)
    min_idx = np.argmin(y) + 1
    min_val = round(np.min(y), 2)
    plt.plot(max_idx, max_val, marker='^')
    plt.plot(min_idx, min_val, marker='v')
    plt.annotate('({}, {})'.format(max_idx, max_val), (max_idx, max_val))
    plt.annotate('({}, {})'.format(min_idx, min_val), (min_idx, min_val))
    plt.savefig('log/{}.png'.format(filename))
    plt.close()