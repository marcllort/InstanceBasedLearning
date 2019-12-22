import scipy
from scipy import stats
import numpy
import sklearn
import sklearn.datasets
from sklearn.model_selection import KFold
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as pyplot





# Carregem datasets

digits = sklearn.datasets.load_digits()

X = digits.data
Y = digits.target

# print X.shape, Y.shape
# print digits.DESCR


# 1. Analitzar el conjunt de dades de digits (Digits Data Set)
valorTest = 3

# Numero de mostres per classe
unique, counts = numpy.unique(Y, return_counts=True)
print(dict(zip(unique, counts)))

# Calcul de la mitjana
mitjana = numpy.mean(X[Y == valorTest, :], axis=0)
print("Mitjana ", mitjana)
pyplot.imshow(mitjana.reshape(8, 8), interpolation="none", cmap="Greys")
pyplot.title("Mitjana: " + str(valorTest))
pyplot.show()

# Calcul de la desviacio tipica
std = numpy.std(X, axis=0)
print("Desviacio tipica: ", std)
pyplot.imshow(std.reshape(8, 8), interpolation="none", cmap="Greys")
pyplot.title("Desviacio tipica")
pyplot.show()

# Opcional: imshow de valorTest
pyplot.imshow(X[valorTest].reshape(8, 8), interpolation="none", cmap="Greys")
pyplot.title("Digit: " + str(Y[valorTest]))
pyplot.show()


# 2. Divisio amb train i test i normalitzacio de les dades

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.30, train_size=0.70)
X_train_norm = scipy.stats.zscore(X_train, axis=1, ddof=1)
X_test_norm = scipy.stats.zscore(X_test, axis=1, ddof=1)


# 3. Projeccio en diferents components principals

# Descomposem les dades amb PCA
pca = PCA(n_components=2).fit_transform(X_train_norm)
pyplot.scatter(pca[:, 0], pca[:, 1], c=Y_train)
pyplot.title("PCA")
pyplot.show()

# Descomposem les dades amb Truncated SVD
svd = TruncatedSVD(n_components=2).fit_transform(X_train_norm)
pyplot.scatter(svd[:, 0], svd[:, 1], c=Y_train)
pyplot.title("Truncated SVD")
pyplot.show()

# Descomposem les dades amb KernelPCA
kernelpca = KernelPCA(n_components=2).fit_transform(X_train_norm)
pyplot.scatter(kernelpca[:, 0], kernelpca[:, 1], c=Y_train)
pyplot.title("Kernel PCA")
pyplot.show()



# 4. Validacio creuada per estimar el nombre optim de veins K
weight = "uniform"
#weight = "distance"
n_splits = 10
min_neighbours = 1
max_neighbours = 11
values = []

for i in range(min_neighbours, max_neighbours):
    values.append(i)

def get_folds(n_splits):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    folds = list(kfold.split(X))
    return folds


def get_predicted(train_X, train_Y, test_X, n_neighbours, weight):
    knearest = KNeighborsClassifier(n_neighbors=n_neighbours, weights=weight)
    model = knearest.fit(train_X, train_Y)
    predicted_y = model.predict(test_X)
    return predicted_y


def compute_test(n_splits, min_neighbours, max_neighbours, weight):
    # Veins i dimensions optimes
    folds = get_folds(n_splits)
    scores = []

    # Proves amb numero de veins variant
    for i in range(min_neighbours, max_neighbours):
        success = 0
        total = 0

        for j in range(len(folds)):
            train, test = folds[j]

            x_train = [X[k] for k in train]
            x_test = [X[k] for k in test]
            y_train = [Y[k] for k in train]
            y_test = [Y[k] for k in test]

            # Prediccio en base al classificador
            predicted_y = get_predicted(x_train, y_train, x_test, i, weight)

            for k in range(len(y_test)):
                total = total + 1
                if y_test[k] == predicted_y[k]:
                    success = success + 1

        scores.append(float(success) / (float(total)) * 100)
    return scores


scores = compute_test(n_splits, min_neighbours, max_neighbours, weight)

# Resultats
fig = pyplot.figure()
ax = fig.add_subplot(111)
ax.plot(values, scores, color='red', linewidth=2)
ax.set_xlim(min_neighbours, max_neighbours)
pyplot.title(weight + " using SVD")
pyplot.xlabel("Neighbours")
pyplot.ylabel("Accuracy")
pyplot.show()
