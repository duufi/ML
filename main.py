import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def createDataSets(filename):
    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, 6].values # at bats
    y = y = dataset.iloc[:, 7].values #runs scored
    from sklearn.linear_model import LinearRegression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = np.reshape(X_train, (-1, 1))
    X_test = np.reshape(X_test, (-1, 1))
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    X_value = np.array(704).reshape(-1, 1)
    predicted_y = regressor.predict(X_value)
    print(f"The predicted Y value for x=704 is {predicted_y[0]}")

    predicted = regressor.predict(X_test)
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.xlabel('At Bats')
    plt.ylabel('Runs')
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filename = "Ichiro.csv"
    createDataSets(filename)







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
