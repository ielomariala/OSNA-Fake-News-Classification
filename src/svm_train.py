from sklearn.ensemble import AdaBoostClassifier
from dataloader import DataLoader
from sklearn.metrics import confusion_matrix



def main():
    dloader = DataLoader('data')
    (train_X, valid_X, train_y, valid_y) = dloader.get_train_set()
    clf = AdaBoostClassifier()
    clf.fit(train_X, train_y)
    print(clf.score(valid_X, valid_y))
    pred = clf.predict(valid_X)
    print(confusion_matrix(valid_y , pred, labels=clf.classes_))


if __name__ == '__main__':
    main()