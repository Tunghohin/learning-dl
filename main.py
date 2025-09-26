from nets.LinearRegression.data import load_data, prepare_data
from nets.LinearRegression import LinearRegression
from utils.train import train
from utils.evaluate import evaluate

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    train_loader = prepare_data(X_train, y_train)
    test_loader = prepare_data(X_test, y_test)

    model = LinearRegression()

    train(model, train_loader, epochs=100, lr=0.01)
    avg_loss = evaluate(model, test_loader)

    print(f'Evaluation Loss: {avg_loss:.4f}')
    
