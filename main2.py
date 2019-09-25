# In[]
import numpy as np  # Biblioteca de manipulacao de arrays Numpy



class MyAdaline(object):
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01, erro_delta_minimo=0.001):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.random.rand(no_of_inputs + 1)
        self.weights[0] = -np.pi
        self.erro_delta_minimo = erro_delta_minimo

    # Modficar adaline para fazer ...
    def predict(self, inputs):
        # multipla as entradas pelo pesos
        return np.sum(np.dot(inputs, self.weights[1:]) + self.weights[0])

    def train(self, training_inputs, labels):
        erroAnterior = 0
        for n in range(self.threshold):
            print("Treianmento ", n)
            erroMedio = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                erro = 0.5 * ((label - prediction)**2)
                erroMedio += erro/(len(labels))
                self.weights[1:] += self.learning_rate * erro * inputs
                self.weights[0] += self.learning_rate * erro
                
            print("Erro Medio ", erroMedio)
            print("Variacao do erro:")
            print((erroMedio - erroAnterior))
            if abs((erroMedio - erroAnterior)) <= self.erro_delta_minimo:
                break
            else:
                erroAnterior = erroMedio

    def trainBatch(self, training_inputs, labels):
        erroAnterior = 0
        for n in range(self.threshold):
            print("Treianmento ", n)
            erroMedio = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                erro = 0.5 * ((label - prediction)**2)
                #print("previ: ", prediction, " o esperado foi: ", label)
                #print("erro foi: ", erro)
                erroMedio += erro/(len(labels))
            print("Erro Medio ", erroMedio)
            print("Variacao do erro:")
            print((erroMedio - erroAnterior))
            if abs((erroMedio - erroAnterior)) <= self.erro_delta_minimo:
                print("eqm minimo")
            else:
                print("Pesos no teste")
                self.printPesos()
                print("inputs:")

                for inputs in training_inputs:
                 #   print(inputs)
                    self.weights[1:] += self.learning_rate * \
                        erroMedio * (inputs)
                  #  self.printPesos()
                self.weights[0] += self.learning_rate * erroMedio
                #print("Pesos corrigidos")
                # self.printPesos()
                erroAnterior = erroMedio

    def printPesos(self):
        print(self.weights)



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
        for i in range(self.threshold):
            erroMedio = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                erro = (label - prediction)
                self.weights[1:] += self.learning_rate * \
                    erro * inputs
                self.weights[0] += self.learning_rate * erro
                erroMedio += erro/len(labels)
            if erroMedio == 0:
                print("Treinamento finalido em ", i, 'epocas')
                break

    def trainBatch(self, training_inputs, labels):
        for i in range(self.threshold):
            erroMedio = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                erro = (label - prediction)
                self.weights[1:] += self.learning_rate * \
                    erro * inputs
                self.weights[0] += self.learning_rate * erro
                erroMedio += erro/len(labels)
            if erroMedio == 0:
                print("Treinamento finalido em ", i, 'epocas')
                break



class MLPCamada(object):
    def __init__(self,no_neuronios=2,no_entradas=2):
        self.weights = np.random.rand(no_entradas + 1, no_neuronios)
        self.no_neuronios = no_neuronios
        self.no_entradas = no_entradas

    def predict(self, inputs):
        # multipla as entradas pelo pesos
        lista = []
        for i in range(self.no_neuronios):
            neuron = self.weights[:,i]
            print(neuron)
            res = np.sum(np.dot(inputs, neuron[1:]) + neuron[0]) 
            lista.append(res)
        
        return np.array(lista)

    def printPesos(self):
        print(self.weights)

m = MLPCamada()
m.printPesos()
a = np.array((0,1))
print(m.predict(a))
    


class MyMLP(object):
    def __init__(self, no_of_inputs,no_camadas=2,no_saidas=1, threshold=100, learning_rate=0.01, erro_delta_minimo=0.001):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.random.rand(no_of_inputs + 1)
        self.erro_delta_minimo = erro_delta_minimo

    # Modficar adaline para fazer ...
    def predict(self, inputs):
        # multipla as entradas pelo pesos
        return np.sum(np.dot(inputs, self.weights[1:]) + self.weights[0])

    def train(self, training_inputs, labels):
        erroAnterior = 0
        for n in range(self.threshold):
            print("Treianmento ", n)
            erroMedio = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                erro = 0.5 * ((label - prediction)**2)
                erroMedio += erro/(len(labels))
                self.weights[1:] += self.learning_rate * erro * inputs
                self.weights[0] += self.learning_rate * erro
                
            print("Erro Medio ", erroMedio)
            print("Variacao do erro:")
            print((erroMedio - erroAnterior))
            if abs((erroMedio - erroAnterior)) <= self.erro_delta_minimo:
                break
            else:
                erroAnterior = erroMedio

    def trainBatch(self, training_inputs, labels):
        erroAnterior = 0
        for n in range(self.threshold):
            print("Treianmento ", n)
            erroMedio = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                erro = 0.5 * ((label - prediction)**2)
                #print("previ: ", prediction, " o esperado foi: ", label)
                #print("erro foi: ", erro)
                erroMedio += erro/(len(labels))
            print("Erro Medio ", erroMedio)
            print("Variacao do erro:")
            print((erroMedio - erroAnterior))
            if abs((erroMedio - erroAnterior)) <= self.erro_delta_minimo:
                print("eqm minimo")
            else:
                print("Pesos no teste")
                self.printPesos()
                print("inputs:")

                for inputs in training_inputs:
                 #   print(inputs)
                    self.weights[1:] += self.learning_rate * \
                        erroMedio * (inputs)
                  #  self.printPesos()
                self.weights[0] += self.learning_rate * erroMedio
                #print("Pesos corrigidos")
                # self.printPesos()
                erroAnterior = erroMedio

    def printPesos(self):
        print(self.weights)

#%%
