import numpy as np  # Biblioteca de manipulacao de arrays Numpy

# Construindo Adaline -- minha base


class Adaline(object):
    def __init__(self, eta=0.001, epoch=100):
        self.eta = eta
        self.epoch = epoch

    def fit(self, X, y):
        np.random.seed(16)
        self.weight_ = np.random.uniform(-1, 1, X.shape[1] + 1)
        self.error_ = []

        cost = 0
        for _ in range(self.epoch):

            output = self.activation_function(X)
            error = y - output

            self.weight_[0] += self.eta * sum(error)
            self.weight_[1:] += self.eta * X.T.dot(error)

            cost = 1./2 * sum((error**2))
            self.error_.append(cost)

        return self

    def net_input(self, X):
        """Calculo da entrada z"""
        return np.dot(X, self.weight_[1:]) + self.weight_[0]

    def activation_function(self, X):
        """Calculo da saida da funcao g(z)"""
        return self.net_input(X)

    def predict(self, X):
        """Retornar valores binaros 0 ou 1"""
        return np.where(self.activation_function(X) >= 0.0, 1, -1)

# Percepetron -- minha base


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        # multipla as entradas pelo pesos
        #
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * \
                    (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


class MyPercetron(object):
    def __init__(self, no_entradas, no_saidas, threshold=100, learning_rate=0.01, ativicacao=1):
        self.vetorPesos = np.random.rand(no_entradas, no_saidas) * 2 - 1
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.ativicacao = ativicacao
        self.no_entradas = no_entradas
        self.no_saidas = no_saidas

    # Entrada com um vetor de input
    # sai um vetor de saida com a predicao de cada neuronio
    def predict(self, entrada):
        predicao = []
        for vetor in self.vetorPesos:
            predicao.append(self.predictNeuronio(entrada, vetor))
        return predicao

    def predictNeuronio(self, entrada, pesos):
        summation = np.dot(entrada, pesos)
        if summation >= 0:  # self.ativicacao:
            return 1
        else:
            return 0

    def train(self, entradas, labels):
        for t in range(self.threshold):
            for entrada, label in zip(entradas, labels):
                # para cada entrada pego uma entrada do neuronio
                predicao = self.predict(entrada)
                predicao = np.array(predicao)
                for ypredict, ylabel, pos in zip(predicao, label, range(0, self.no_saidas)):
                    if ylabel != ypredict:
                        # ajusta os pesos
                        erro = ylabel - ypredict
                        # vezes a entrada do neuronio
                        self.vetorPesos[pos, :] = self.vetorPesos[pos,
                                                                  :] + (self.learning_rate * erro)
                        if ylabel > ypredict:
                            # sobe
                            print("ajust pra cima")
                        if ylabel < ypredict:
                            # desce
                            print("ajust pra baixo")
                        self.printVetorPesos()

    def printVetorPesos(self):
        for v in self.vetorPesos:
            print(v)


redeAdaline = Adaline(eta=0.01)
redePercetron = Perceptron(1)

myPercepetron = MyPercetron(2, 2)
myPercepetron.printVetorPesos()
vetorEntrada = [1, 1]
resultado = myPercepetron.predict(vetorEntrada)
print(resultado)

matrizX = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
matrizY = np.array([(0, 0), (0, 1), (0, 1), (1, 1)])

myPercepetron.train(matrizX, matrizY)
