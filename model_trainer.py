import joblib
import os
import numpy as np

from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from settings import BINARY_MODEL


class ClassifierTrainer:
    def __init__(self):
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                            "Naive Bayes", "QDA"]
        self.classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=2, degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1,
                decision_function_shape='ovr', random_state=None),
            SVC(gamma=2, C=1, probability=True),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

    @staticmethod
    def convert_str_array(array_str):
        list_str = array_str.replace("  ", " ").replace(" ", ",")
        last_comma = list_str.rfind(",")
        f_list_str = list_str[:last_comma] + list_str[last_comma + 1:]
        converted_array = np.array(literal_eval(f_list_str))

        return converted_array

    def train_best_model(self, model_name, x_data, y_data):

        x_train, x_test, y_train, y_test = \
            train_test_split(x_data, y_data, test_size=.3, random_state=42)
        best_clf = self.classifiers[self.model_names.index(model_name)]
        # print(f"Best Y data: {y_data}")
        best_clf.fit(x_data, y_data)
        score = best_clf.score(x_test, y_test)
        print(score)
        joblib.dump(best_clf, BINARY_MODEL)
        print(f"Successfully saved in {BINARY_MODEL}")

        return

    def test_several_models(self, x_data, y_data):
        x_train, x_test, y_train, y_test = \
            train_test_split(x_data, y_data, test_size=.3, random_state=42)

        model_names = []
        accuracies = []

        for name, clf in zip(self.model_names, self.classifiers):
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            print(f"model:{name}, score:{score}")
            model_names.append(name)
            accuracies.append(score)

        return model_names, accuracies


if __name__ == '__main__':
    # ClassifierTrainer().test_several_models(x_data=[], y_data=[])
    ClassifierTrainer().train_best_model(model_name="RBF SVM", x_data=[], y_data=[])
