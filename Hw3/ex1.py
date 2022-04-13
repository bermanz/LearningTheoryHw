
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from p_tqdm import t_map

def get_dataset(set_type):
    return pd.read_csv(fr'Hw3\{set_type}.csv', names=["$x_1$", "$x_2$", "y"])

def plot_set(set, title):
    plt.figure()
    sns.scatterplot(data=set, x="$x_1$", y="$x_2$", hue="y", palette="pastel")
    plt.title(title)
    
def poly_eval(prod, q):
    return (1 + prod) ** q

def rbf_eval(dist_squared, sigma):
    exp_arg = -dist_squared / (2*sigma**2)
    return np.exp(exp_arg)

def get_poly_kernel(x, q):
    gram_mat = x @ x.T
    return poly_eval(gram_mat, q)

def get_rbf_kernel(x, sigma):
    m = x.shape[0]
    x_rep = np.tile(x, [m, 1]) 
    x_reshaped = x_rep.reshape((m, m, 2)) # each column stands for one coordinate, 3rd channel is the dimension
    x_diffs = x_reshaped - x_reshaped.transpose((1, 0, 2))
    x_dists_squared = (x_diffs ** 2).sum(axis=2)
    return rbf_eval(x_dists_squared, sigma)

def get_kernel(set, kernel_type, hyper_param):
    x = set[["$x_1$", "$x_2$"]].values
    if kernel_type=="polynomial":
        return get_poly_kernel(x, q=hyper_param)
    elif kernel_type=="rbf":
        return get_rbf_kernel(x, sigma=hyper_param)


def train_perceptron(train_set, kernel_type="polynomial", hyper_param=3, max_iter=99):
    
    y = train_set["y"].values.astype(int)
    m = len(y)
    y_hat = np.zeros(m, dtype=int)
    alpha = np.zeros(m, dtype=int)
    K = get_kernel(train_set, kernel_type, hyper_param)
    T = 0

    # iterate for T iterations:
    while not (y_hat==y).all():        
        for i in range(m):
            y_hat[i] = np.sign(alpha @ K.T[:, i])
            if T == 0 and i == 0: # first iteration after initialization - alpha will be a float (fix by assigning y instead)
                alpha[i] = y[i]
            else:
                alpha[i] += 0.5 * (y[i]- y_hat[i])
        T += 1
        if T > max_iter:
            T = None
            break
    return alpha, T


def inspect_iter_vs_hyper_param(train_set, kernel_type="polynomial"):
    if kernel_type == "polynomial":
        hyper_grid = np.arange(start=1, stop=11)
        param_name = "order"
    elif kernel_type == "rbf":
        hyper_grid = np.logspace(-5, 5, base=2, num=10)
        param_name = "$sigma$"

    train_func = partial(train_perceptron, train_set, kernel_type)
    iters = np.zeros_like(hyper_grid)
    res = t_map(train_func, hyper_grid)
    iters = np.asarray([x[1] for x in res])
    _, ax = plt.subplots()
    ax.plot(hyper_grid, iters, '-o')
    if kernel_type == "rbf":
        ax.set_xscale("log")
    ax.set_xlabel(param_name)
    ax.set_ylabel("T")
    ax.set_title(f"{kernel_type.capitalize()} Kernel")
    ax.grid()

def main():
    train_set = get_dataset("train")    
    plot_set(train_set, title="Training Set")
    # alpha, T = train_perceptron(train_set)
    for kernel in ["polynomial", "rbf"]:
        inspect_iter_vs_hyper_param(train_set, kernel_type=kernel)
    
    plt.show()
    a = 1

if __name__=="__main__":
    main()
