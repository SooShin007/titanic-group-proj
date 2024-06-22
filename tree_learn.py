# Paige Hicks and Soo Shin
# tree_learn.py
# This program cleans the titanic data and classifies the data in 2 different ways
# with RandomForestClassification and K-Nearest-Neighbor classification
# Then the program proceeds to produce learning curve visuals for both classification methods
# Also produces the first tree in the the random forest

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt


def print_output(evaluation):
    """
    Helper function for test_model
    Formats Prints to Look pretty
    :param evaluation:
    :return: None
    """
    print("The precision ranges from", evaluation[0][0], "to", evaluation[0][1])
    print("The recall ranges from", evaluation[1][0], "to", evaluation[1][1])
    print("The fscore ranges from", evaluation[2][0], "to", evaluation[2][1])
    print("The support ranges from", evaluation[3][0], "to", evaluation[3][1])
    print()


def test_model(prediction, Y):
    """
    This function prints out metrics for our classification
    :param prediction: the predicted values of our classifier
    :param Y: what we are checking against
    :return: None
    """
    # Get misclassifications for a quick idea of how well improvements are doing
    err = 0
    for i in range(0, len(prediction)):
        if prediction[i] != Y[i]:
            print(i)
            err += 1
    print("Misclassifications:", err)
    print()
    # Print out values from precision_recall_fscore_support
    print("precision_recall_fscore_support of cleaned data:")
    evaluation = precision_recall_fscore_support(Y, prediction)
    print_output(evaluation)


def clean_data(X):
    """
    This function handles all of the cleaning of our data
    :param X: Parsed columns of attributes
    :return: Cleaned X
    """
    for i in range(0, len(X['Cabin'])):
        cabin = X['Cabin'][i]
        if str(cabin) == "nan":
            X.iloc[i, X.columns.get_loc('Cabin')] = 0
        elif len(cabin) > 1 and len(cabin) <= 4:
            numVal = int(cabin[1:])
            charVal = cabin[:1]
            numVal += (int(ord(charVal)) - 64) * 100  # A = +100, B = +200, etc
            X.iloc[i, X.columns.get_loc('Cabin')] = numVal
        elif len(cabin) == 1:
            X.iloc[i, X.columns.get_loc(
                'Cabin')] = 0  # better results if we set all bad cabin #s to 0 rather than trying to parse single characters
        else:
            cabins = cabin.split()
            cabinVal = 0
            while len(cabins[0]) == 1:
                cabins = cabins[1:]
            for cabin in cabins:
                # ex 23 C25 C27
                numVal = int(cabin[1:])
                charVal = cabin[:1]
                cabinVal += (int(
                    ord(charVal)) - 64) * 100  # A = +100, B = +200, etc
            X.iloc[i, X.columns.get_loc('Cabin')] = cabinVal

    # Sex cleaning
    X['Sex'] = [0 if sex == 'male' else 1 for sex in X['Sex']]

    # Ticket cleaning
    X['Ticket'] = [len(ticket) for ticket in X['Ticket']]

    # Embarked cleaning
    # the location the passenger embarked from, ranked by distance
    # highest value is the last place visited
    for i in range(0, len(X['Embarked'])):
        embark = X['Embarked'][i]
        if embark == "S":
            X.iloc[i, X.columns.get_loc('Embarked')] = 3
        if embark == "C":
            X.iloc[i, X.columns.get_loc('Embarked')] = 2
        if embark == "Q":
            X.iloc[i, X.columns.get_loc('Embarked')] = 1
        else:
            X.iloc[i, X.columns.get_loc(
                'Embarked')] = 0  # nan, pick the middle value if we don't know

    # Name Cleaning
    names = []
    for name in X['Name']:
        ordName = 0
        for letter in name:
            ordName += ord(letter)
        names.append(ordName)
        ordName = 0
    X['Name'] = names

    # Clean remaining data
    X = X.fillna(0)
    return X


def learning_plot(train_sizes, train_scores, test_scores, fit_times, fname):
    """
    This function is a helper function to plot the learning curve
    :param train_sizes:
    :param train_scores:
    :param test_scores:
    :param fit_times:
    :param fname: filename to be saved at
    :return: None
    """
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.savefig(fname)


def main():
    """
    This is our main function
    It does both classification methods
    and calls our test model function for metrics
    Lastly it produces visuals
    :return: None
    """
    train = pd.read_csv("train.csv")

    # Divide input data X from labeled values to predict Y
    X = train.loc[:, 'Pclass':]
    Y = train.loc[:, 'Survived']

    # clean the data we will use
    X = clean_data(X)

    # testing with k-NN
    # distance is important here!  otherwise it misclassifies a lot
    neigh = KNeighborsClassifier(n_neighbors=15, weights="distance")
    neigh = neigh.fit(X, Y)
    predictionKNN = neigh.predict(X)

    # testing with Random Forest Classifier
    clfRandFor = RandomForestClassifier(n_estimators=20)
    clfRandFor.fit(X, Y)
    predrand = clfRandFor.predict(X)

    # plot the learning curve for K-NN
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(neigh, X, Y, train_sizes=np.linspace(.1, 1.0, 5), cv=None, return_times=True)
    learning_plot(train_sizes, train_scores, test_scores, fit_times, "KNN Learning Curve")

    # plot the learning curve for Random Forest
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clfRandFor, X, Y,
                                                                          train_sizes=np.linspace(.1, 1.0, 5), cv=None,
                                                                          return_times=True)
    learning_plot(train_sizes, train_scores, test_scores, fit_times, "Random Forest Learning Curve")

    # Plotting and saving the first tree produced by the random forest generator
    fig = plt.figure(figsize=(80, 80))
    _ = tree.plot_tree(clfRandFor.estimators_[0], feature_names=X.columns, filled=True)
    fig.savefig("firstTree.png")

    # Printing metrics
    print("=============== K-NN results ===============")
    test_model(predictionKNN, Y)
    print()
    print("=============== Random Forest results ===============")
    test_model(predrand, Y)


main()

