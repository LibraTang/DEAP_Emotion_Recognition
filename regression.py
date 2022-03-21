import numpy as np
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor

with open('out\data_training.npy', 'rb') as fileTrain:
    data_train = np.load(fileTrain)

with open('out\label_training.npy', 'rb') as fileTrainL:
    data_train_label = np.load(fileTrainL)

data_train = normalize(data_train)
# Z = np.ravel(Y[:, [1]])

Arousal_Train = np.ravel(data_train_label[:, [0]])
Valence_Train = np.ravel(data_train_label[:, [1]])
Domain_Train = np.ravel(data_train_label[:, [2]])
Like_Train = np.ravel(data_train_label[:, [3]])

with open('out\data_testing.npy', 'rb') as fileTest:
    data_test = np.load(fileTest)

with open('out\label_testing.npy', 'rb') as fileTestL:
    data_test_label = np.load(fileTestL)

data_test = normalize(data_test)
# L = np.ravel(N[:, [1]])

Arousal_Test = np.ravel(data_test_label[:, [0]])
Valence_Test = np.ravel(data_test_label[:, [1]])
Domain_Test = np.ravel(data_test_label[:, [2]])
Like_Test = np.ravel(data_test_label[:, [3]])


def test(data_test, data_test_label, model):
    """
    arguments:  data_test: testing dataset
                data_test_label: testing dataset label
                model: scikit-learn model

    return:     void
    """
    output = model.predict(data_test[0:78080:32])
    label = data_test_label[0:78080:32]

    k = 0
    l = 0

    for i in range(len(label)):
        k = k + (output[i] - label[i]) * (output[i] - label[i])  # square difference

        # a good guess
        if output[i] > 5 and label[i] > 5:
            l = l + 1
        elif output[i] < 5 and label[i] < 5:
            l = l + 1

    print("classification accuracy:", l / len(label), l, len(label))


# start emotion recognition
Val_R = RandomForestRegressor(n_estimators=512, n_jobs=6)
Val_R.fit(data_train[0:468480:32], Valence_Train[0:468480:32])
test(data_test, Valence_Test, Val_R)

Aro_R = RandomForestRegressor(n_estimators=512, n_jobs=6)
Aro_R.fit(data_train[0:468480:32], Arousal_Train[0:468480:32])
test(data_test, Arousal_Test, Aro_R)

Dom_R = RandomForestRegressor(n_estimators=512, n_jobs=6)
Dom_R.fit(data_train[0:468480:32], Domain_Train[0:468480:32])
test(data_test, Domain_Test, Dom_R)

Lik_R = RandomForestRegressor(n_estimators=512, n_jobs=6)
Lik_R.fit(data_train[0:468480:32], Like_Train[0:468480:32])
test(data_test, Like_Test, Lik_R)
