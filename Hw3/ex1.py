
from abc import abstractmethod, ABC
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from p_tqdm import t_map


class Perceptron(ABC):

    def __init__(self) -> None:
        self.train_set = self._get_dataset("train")
        self.alpha = None

    @staticmethod
    def _get_dataset(set_type):
        return pd.read_csv(fr'Hw3\{set_type}.csv', names=["$x_1$", "$x_2$", "y"])

    def plot_dataset(self):
        plt.figure()
        sns.scatterplot(data=self.train_set, x="$x_1$", y="$x_2$", hue="y", palette="pastel")
        plt.title("Training Set")
    
    @abstractmethod
    def _activate(self, arg):
        pass

    @abstractmethod
    def _get_kernel(self):
        pass

    @abstractmethod
    def _get_kernel_prod(self, x_test):
        pass

    def get_features(self):
        return self.train_set[["$x_1$", "$x_2$"]].values

    def get_labels(self):
        return self.train_set["y"].values.astype(int)


    def train(self, max_iter=99):
        
        y = self.get_labels()
        m = len(y)
        y_hat = np.zeros(m, dtype=int)
        alpha = np.zeros(m, dtype=int)
        K = self._get_kernel()
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
                return None

        self.alpha = alpha
        return T
    
    def predict(self, test_set):
        x_test = test_set[["$x_1$", "$x_2$"]].values
        x_kernel_prod = self._get_kernel_prod(x_test)
        y_hat = x_kernel_prod @ self.alpha
        return np.sign(y_hat)

    def eval(self, test_set):
        y_hat = self.predict(test_set)
        res = pd.DataFrame(data=np.hstack((test_set.values, np.atleast_2d(y_hat).T)), columns=[*test_set.columns, "y_hat"])
        return res


class PolyKernel(Perceptron):

    def __init__(self, q:int) -> None:
        super().__init__()
        self.q = q

    def _activate(self, arg):
        return (1 + arg) ** self.q
        
    def _get_kernel(self):
        x = self.get_features()
        gram_mat = x @ x.T
        return self._activate(gram_mat)

    def _get_kernel_prod(self, x_test):
        x_train = self.get_features()
        x_prod = x_test @ x_train.T
        return self._activate(x_prod)


class RbfKernel(Perceptron):

    def __init__(self, sigma:float) -> None:
        super().__init__()
        self.sigma = sigma

    def _activate(self, arg):
        exp_arg = -arg / (2*self.sigma**2)
        return np.exp(exp_arg)

    def _get_kernel(self):
        x = self.get_features()
        m = x.shape[0]
        x_rep = np.tile(x, [m, 1]) 
        x_reshaped = x_rep.reshape((m, m, 2)) # each column stands for one coordinate, 3rd channel is the dimension
        x_diffs = x_reshaped - x_reshaped.transpose((1, 0, 2))
        x_dists_squared = (x_diffs ** 2).sum(axis=2)
        return self._activate(x_dists_squared)

    def _get_kernel_prod(self, x_test):
        x_train = self.get_features()
        x_diffs = np.repeat(x_test, len(x_train), axis=0) - np.tile(x_train, [len(x_test), 1])
        x_squared_diffs = (x_diffs ** 2).sum(axis=1).reshape((len(x_test), -1))
        return self._activate(x_squared_diffs)


def main():
    # Plot training set:
    perceptron = PolyKernel(q=3)
    perceptron.plot_dataset()
    
    # experiment to get the correct hyper-parameters for an ERM algorithm, and show the results over the test set:
    train_set = perceptron._get_dataset("train")
    test_set = perceptron._get_dataset("test")
    _, ax_T = plt.subplots(2, 1)
    _, ax_test = plt.subplots(1, 2)

    for j, kernel_type in enumerate(["polynomial", "rbf"]):

        # Find optimal T for arbitrary hyper-parameter values:
        if kernel_type == "polynomial":
            perceptron_prot = PolyKernel
            hyper_grid = np.arange(start=1, stop=11)
            param_name = "order"
        elif kernel_type == "rbf":
            perceptron_prot = RbfKernel
            hyper_grid = np.logspace(-5, 5, base=2, num=10)
            param_name = "$sigma$"
        iters = []
        for param in hyper_grid:
            perceptron = perceptron_prot(param)
            iters.append(perceptron.train())
        ax_T[j].plot(hyper_grid, iters, '-o')
        if kernel_type == "rbf":
            ax_T[j].set_xscale("log")
        ax_T[j].set_xlabel(param_name)
        ax_T[j].set_ylabel("T")
        ax_T[j].set_title(f"{kernel_type.capitalize()} Kernel")
        ax_T[j].grid()

        # Choose one hyper-param and show performance on test set:
        hyper_grid = [hyper_param for iter, hyper_param in zip(iters, hyper_grid) if not iter is None]
        hyper_param = np.round(np.random.choice(hyper_grid, 1)[0], decimals=2)
        perceptron = perceptron_prot(hyper_param)
        T = perceptron.train()
        train_res = perceptron.eval(train_set)
        train_res["set"] = "train"
        test_res = perceptron.eval(test_set)
        test_res["set"] = "test"
        res = pd.concat([train_res, test_res], ignore_index=True)
        res_err = res[res["y"] != res["y_hat"]]
        res.drop(res_err.index, inplace=True)
        sns.scatterplot(ax=ax_test[j], data=res, x="$x_1$", y="$x_2$", hue="y_hat", style="set", palette="pastel")
        sns.scatterplot(ax=ax_test[j], data=res_err, x="$x_1$", y="$x_2$", hue="y_hat", style="set", palette=["green"], legend=False, markers={"train":"o", "test":"X"})
        ax_test[j].set_title(f"{kernel_type.capitalize()} kernel with {param_name}={hyper_param}")

    
    plt.show()


if __name__=="__main__":
    main()
