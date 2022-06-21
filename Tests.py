if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from Perceptron import Perceptron
    from sklearn import metrics
    import numpy as np

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def precision(y_true,y_pred): 
        #tp/tp+fp
        tp=0
        for gt,pred in zip(y_true,y_pred):
            if gt==1 and pred==1:
                tp+=1
        fp=0
        for gt,pred in zip(y_true,y_pred):
            if gt==0 and pred==1:
                fp+=1
        prec=tp/(tp+fp)
        return prec       
    def recall(y_true,y_pred): 
        #tp/tp+fn
        tp=0
        for gt,pred in zip(y_true,y_pred):
            if gt==1 and pred==1:
                tp+=1
        fn=0
        for gt,pred in zip(y_true,y_pred):
            if gt==1 and pred==0:
                fn+=1
        rec=tp/(tp+fn)
        return rec                    

    #creating a dataset with 2 features and total samples 200
    #return X as generated sample and y as integer labels for cluster membership
    X, y = datasets.make_blobs(
        n_samples=200, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )


    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))
    print("Perceptron classification PRecision", precision(y_test, predictions))
    print("Perceptron classification Recall", recall(y_test, predictions))
    print("from sklearn precision score is : ",metrics.precision_score(y_test,predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    # x0_1 = np.amin(X_train[:, 0])
    # x0_2 = np.amax(X_train[:, 0])

    # x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    # x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    # ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    # ymin = np.amin(X_train[:, 1])
    # ymax = np.amax(X_train[:, 1])
    # ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()