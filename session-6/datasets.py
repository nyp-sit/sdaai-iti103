from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

def load_scaled_boston_data():
    boston = load_boston()
    X = boston.data

    ## Never do this in real ML project. Here we fit the scalar to entire dataset, aargh
    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, boston.target