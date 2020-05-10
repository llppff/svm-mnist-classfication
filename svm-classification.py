import getdata

# load torch
import torchvision

# other utilities
# import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix




# %% Main function for MNIST dataset
if __name__ == '__main__':


    # Load Training Data & Testing Data
    train_data, train_label, test_data, test_label, train_nums, test_nums = getdata.run()


    train_data = train_data / 255
    test_data = test_data / 255
    training_features = train_data.reshape(train_nums, -1)
    test_features = test_data.reshape(test_nums, -1)

    # Training SVM
    print('------Training and testing SVM------')
    clf = svm.SVC(C=5, gamma=0.05, max_iter=100)
    clf.fit(training_features, train_label)

    # Test on test data
    test_result = clf.predict(test_features)
    precision = sum(test_result == test_label) / test_label.shape[0]
    print('Test precision: ', precision)

    # Test on Training data
    train_result = clf.predict(training_features)
    precision = sum(train_result == train_label) / train_label.shape[0]
    print('Training precision: ', precision)

    # Show the confusion matrix
    matrix = confusion_matrix(test_label, test_result)