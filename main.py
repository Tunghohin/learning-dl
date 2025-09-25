import torch
import examples.linear_reg as linreg

if __name__ == '__main__':
    model = linreg.LinearReg()
    (X_train, y_train), (X_test, y_test) = linreg.load_data()
    X_train, y_train = linreg.prepare_data(X_train, y_train)
    X_test, y_test = linreg.prepare_data(X_test, y_test)
    print(model.eval(X_test, y_test))
    model.train(X_train, y_train, epochs=100)
    print(model.eval(X_test, y_test))
    
    
