import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl


class WeakLearner:
    def __init__(self, idx, theta, polarity=0) -> None:
        self.idx = idx
        self.theta = theta
        self.polarity = polarity

    def predict(self, x):
        """returns the prediction over the input x at pixel index idx, given a threshould theta and a polarity"""
        base = (x[:, self.idx] < self.theta).astype(int) * 2 - 1
        return base if self.polarity == 0 else -base


class AdaBoost:
    def __init__(self, T=30, pre_trained=False) -> None:
        self.x_train = np.loadtxt(fname=r"Hw3\MNIST_train_images.csv", delimiter=",")
        self.y_train = np.loadtxt(fname=r"Hw3\MNIST_train_labels.csv", delimiter=",")
        self.T = T
        self.theta_grid = np.arange(256+1)
        if pre_trained:
            self.load_learners()
        else:
            self.learners = {}

    def plot_arbitrary_idx(self):
        idx = np.random.choice(len(self.y_train), 1).item()
        plt.figure(label="MNist_random")
        plt.imshow(
            np.asarray(np.reshape(self.x_train[idx, :], (28, 28))),
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        plt.title(f"index={idx}")

    def get_all_pred_errs(self):
        """calculate the prediction errors of all possible weak-predictors over the training set"""
        all_pred_errs = np.empty((len(self.theta_grid), *self.x_train.shape), dtype=bool)
        for theta in tqdm(self.theta_grid, desc="pre-predicting all weak learners", leave=False): 
            h_theta = WeakLearner(idx=np.arange(self.x_train.shape[-1]), theta=theta)
            sample_pixels_class = h_theta.predict(self.x_train)
            all_pred_errs[theta] = sample_pixels_class != np.tile(
                self.y_train, [self.x_train.shape[1], 1]).T

        return all_pred_errs

    def eval_threshold(self, pred_errs, theta, p_t):
        
        all_polarity_fails = np.asarray([pred_errs.T, ~pred_errs.T])
        all_polarity_expectation = all_polarity_fails @ p_t
        best_idxs = np.argmin(all_polarity_expectation, axis=1)
        best_polarity = np.argmin([all_polarity_expectation[i, best_idxs[i]] for i in [0, 1]])
        best_idx = best_idxs[best_polarity]
        best_err = all_polarity_expectation[best_polarity, best_idx]
        best_cand = {
            "idx": best_idx,
            "theta": theta,
            "polarity": best_polarity,
            "p_err": best_err,
        }
        return best_cand

    def get_best_classifier(self, all_pred_errs, p_t):
        best_classifiers = [
            self.eval_threshold(all_pred_errs[theta], theta, p_t)
            for theta in tqdm(self.theta_grid, desc="Finding best weak classifier", leave=False)
        ]
        best_classifiers_sorted = sorted(best_classifiers, key=lambda x: x["p_err"])
        best_classifier = best_classifiers_sorted.pop(0)
        p_err = best_classifier.pop("p_err")
        return best_classifier, p_err

    def save_learners(self):
        with open(r"Hw3\learners.pkl", "wb") as f:
            pkl.dump(self.learners, f)

    def load_learners(self):
        with open(r"Hw3\learners.pkl", "rb") as f:
            self.learners = pkl.load(f)

    def train(self, save_learners=False):

        m = len(self.y_train)
        p_t = np.full_like(self.y_train, fill_value=1 / m)

        all_pred_errs = self.get_all_pred_errs()

        # iterate for T iterations:
        for t in tqdm(range(self.T), desc="AdaBoost Iterations"):
            h_t_params, epsilon_t = self.get_best_classifier(all_pred_errs, p_t)
            h_t = WeakLearner(**h_t_params)
            alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
            p_t_update_arg = p_t * np.exp(
                -alpha_t * self.y_train * h_t.predict(self.x_train)
            )
            p_t = p_t_update_arg / p_t_update_arg.sum()

            # save parameters:
            self.learners[t] = {"h_t": h_t, "alpha_t": alpha_t}

        if save_learners:
            self.save_learners()

    def predict(self, x_test, t):
        if not self.learners:
            raise Exception("No trained learners are available!")
        elif t not in self.learners:
            raise Exception(f"no learner available for the {t} iteration!")
        weak_predictors = np.asarray(
            [v["h_t"].predict(x_test) for k, v in self.learners.items() if k <= t]
        ).T
        alphas = np.asarray([v["alpha_t"] for k, v in self.learners.items() if k <= t])
        strong_prediction = np.sign(weak_predictors @ alphas)
        return strong_prediction


def main():

    T = 30
    # Plot training set:
    adaboost = AdaBoost(T)
    adaboost.plot_arbitrary_idx()
    adaboost.train()

    # Evaluate AdaBoost:
    plt.figure(label="AdaBoost")
    plt.title("AdaBoost Error Rate Vs Iterations")
    plt.ylabel("# Prediction Errors")
    plt.xlabel("Iteration")

    for phase in ["train", "test"]:
        n_errors = np.empty(T)
        x = np.loadtxt(fname=rf"Hw3\MNIST_{phase}_images.csv", delimiter=",")
        y = np.loadtxt(fname=rf"Hw3\MNIST_{phase}_labels.csv", delimiter=",")
        for t in tqdm(range(T), desc="Predicting "):
            y_hat = adaboost.predict(x, t)
            err = y_hat != y
            n_errors[t] = err.sum()

        plt.plot(np.arange(T) + 1, n_errors, "-o", label=phase)

    plt.legend()
    plt.grid()

    for label in plt.get_figlabels(): 
        plt.figure(label)
        plt.savefig(rf"Hw3\{label}.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
