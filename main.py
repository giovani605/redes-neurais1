# In[]
import numpy as np  # Biblioteca de manipulacao de arrays Numpy

# Construindo Adaline -- minha base


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


# In[]
# Vetor entrada
# vetor resposta
print("Tarefa 3.1")

perceptronOr = Perceptron(2)
percetronAnd = Perceptron(2)

matrizX = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
matrizY = np.array([(0, 0), (0, 1), (0, 1), (1, 1)])
print("Treino Perceptron Or interativo")
perceptronOr.train(matrizX, matrizY[:, 1])
print("Treino Perceptron And interativo")
percetronAnd.train(matrizX, matrizY[:, 0])


perceptronOrBatch = Perceptron(2)
percetronAndBatch = Perceptron(2)
print("Treino Perceptron Or batch")
perceptronOrBatch.trainBatch(matrizX, matrizY[:, 1])
print("Treino Perceptron And batch")
percetronAndBatch.trainBatch(matrizX, matrizY[:, 0])


# %%
print("Teste percetron or")
for row in matrizX:
    print(perceptronOr.predict(row))

print("Teste percetron And")
for row in matrizX:
    print(percetronAnd.predict(row))

# In[]
# gerar dados para o adaline
# garantir q essa func esta certa


def gerarDadosAdaline(qtd):
    lista = []
    a = np.sin(np.pi)
    print(a)

    def corrigir(a):
        if abs(a) < 0.0001:
            return 0
        else:
            return a

    for p in np.linspace(0, 2*np.pi, qtd):
        z = p
        f1 = corrigir(np.sin(z))
        f2 = corrigir(np.cos(z))
        f3 = z
        output = -np.pi + (0.565*f1)+(2.657*f2)+(0.674*f3)
        lista.append([f1, f2, f3, output])
    print('sdasadsa')
    lista = np.array(lista)
    np.random.shuffle(lista)
    return lista[:,:3],lista[:,3]

def testarAdaline(clf,X,y):
    erroMedio = 0
    for row,label in zip(X,y):
        predicao = clf.predict(row)
        erroMedio += abs((label - predicao))
    print("Erro medio do treino: ", erroMedio/len(y))

dadosX,dadosY = gerarDadosAdaline(15)
dadosX

# %%
# Adaline


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


#dadosX = np.random.shuffle(dadosX, DadosY)
# In[] 
# tarefa 3.2
dadosX ,dadosY= gerarDadosAdaline(15)

ada = MyAdaline(3, learning_rate=0.0001, threshold=50)
ada.train(dadosX, dadosY)
testarAdaline(ada,dadosX,dadosY)

    # %%


# In[]
# tarefa 3.3
dadosX ,dadosY= gerarDadosAdaline(15)
ada = MyAdaline(3, learning_rate=0.0001, threshold=50)
ada.trainBatch(dadosX, dadosY)

testarAdaline(ada,dadosX,dadosY)

# In[]
# tarefa 3.4
dadosX,dadosY = gerarDadosAdaline(30)

# dividir dados

####

ada = MyAdaline(3, learning_rate=0.0001, threshold=50)
ada.trainBatch(dadosX, dadosY)

## testar

testarAdaline(ada,dadosX,dadosY)

# In[]
# tarefa 3.5
dadosX,dadosY = gerarDadosAdaline(50)

# dividir dados

####

ada = MyAdaline(3, learning_rate=0.0001, threshold=50)
ada.trainBatch(dadosX, dadosY)

testarAdaline(ada,dadosX,dadosY)