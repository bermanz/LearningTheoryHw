
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class AdaBoost:

    def __init__(self, T=30) -> None:
        self.x_train =  np.loadtxt(fname=r'Hw3\MNIST_train_images.csv', delimiter=',')
        self.y_train =  np.loadtxt(fname=r'Hw3\MNIST_train_labels.csv', delimiter=',')
        self.T=T
        self.learners = []

    def plot_arbitrary_idx(self):
        idx = np.random.choice(len(self.y_train), 1).item()
        plt.imshow(np.asarray(np.reshape(self.x_train[idx, :], (28, 28))), cmap='gray', vmin=0, vmax=255)
        plt.title(f"index={idx}")
    
    def h_theta(self, x, idx, theta, polarity=0):
        """returns the prediction over the input x at pixel index idx, given a threshould theta and a polarity"""
        base = (x[:, idx] < theta).astype(int) * 2 - 1
        return base if polarity == 0 else -base


    def eval_threshold(self, theta, p_t):
        #  based classification:
        sample_pixels_class = self.h_theta(self.x_train, np.arange(784), theta).T
        sample_pixels_fails = sample_pixels_class != np.tile(self.y_train, [self.x_train.shape[1], 1])
        tot_fails = sample_pixels_fails @ p_t
        conj_fails = ~sample_pixels_fails @ p_t  # reflects the failure of conjugate h_theta (with polarity=1)

        comb_fails = np.asarray([tot_fails, conj_fails])
        best_pixels = np.argmin(comb_fails, axis=1)
        best_polarity = np.argmin([comb_fails[i, best_pixels[i]] for i in [0, 1]])
        best_pixel = best_pixels[best_polarity]
        best_err = comb_fails[best_polarity, best_pixel]
        return best_pixel, best_polarity, best_err

    def get_best_classifier(self, p_t):
        best_classifiers = {theta: self.eval_threshold(theta, p_t) for theta in range(256)}
        best_classifiers_sorted = [(k, v) for k, v in sorted(best_classifiers.items(), key=lambda item:item[1][2])]
        best_classifier = best_classifiers_sorted[0]
        best_theta = best_classifier[0]
        best_idx = best_classifier[1][0]
        best_polarity = best_classifier[1][1]
        epsilon = best_classifier[1][2]
        return partial(self.h_theta, idx=best_idx, theta=best_theta, polarity=best_polarity), epsilon
        

    def train(self):
        
        m = len(self.y_train)
        p_t = np.full_like(self.y_train, fill_value=1/m)

        # iterate for T iterations:
        for _ in tqdm(range(self.T), desc="AdaBoost Iterations"):
            h_t, epsilon_t = self.get_best_classifier(p_t)
            alpha_t = 0.5 * np.log((1-epsilon_t) / epsilon_t)
            p_t_update_arg = p_t * np.exp(-alpha_t * self.y_train * h_t(self.x_train))
            p_t = p_t_update_arg / p_t_update_arg.sum()
            self.learners.append(lambda x: alpha_t*h_t(x))

        a = 1

    def predict(self, x_test, t):
        return 

    def eval(self):
        x_test = np.loadtxt(fname=r'Hw3\MNIST_test_images.csv', delimiter=',')
        y_test =  np.loadtxt(fname=r'Hw3\MNIST_test_labels.csv', delimiter=',')
        n_errors = np.empty(range(self.T))
        for t in range(self.T):
            y_hat = self.predict(x_test, t)

        res = pd.DataFrame(data=np.hstack((test_set.values, np.atleast_2d(y_hat).T)), columns=[*test_set.columns, "$\hat{y}$"])
        return res



def main():
    # Plot training set:
    adaboost = AdaBoost()
    # adaboost.plot_arbitrary_idx()
    adaboost.train()

    


if __name__=="__main__":
    main()
