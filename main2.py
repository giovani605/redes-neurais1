# In[]
import numpy as np  # Biblioteca de manipulacao de arrays Numpy
from sklearn.neural_network import MLPClassifier,MLPRegressor


class MLPCamada(object):
    def __init__(self, no_neuronios=2, no_entradas=2):
        # matriz de pesos para cada  neuronio
        self.weights = np.random.rand(no_entradas + 1, no_neuronios)
        self.no_neuronios = no_neuronios
        self.no_entradas = no_entradas
        # Vetor de erro de cada neuronio
        self.no_erro = np.random.rand(no_neuronios)

    def getErroCamada(self):
        return self.no_erro

    def predictFunc(self, x):
        return (1 / (1 + (pow(np.e, -x))))

    def derivadaFunc(self, x):
        return self.predictFunc(x)*(1 - self.predictFunc(x))

    def predict(self, inputs):
        # multipla as entradas pelo pesos
        lista = []
        for i in range(self.no_neuronios):
            neuron = self.weights[:, i]
            res = self.predictFunc(
                np.sum(np.dot(inputs, neuron[1:]) + neuron[0]))
            lista.append(res)

        return np.array(lista)

    def printPesos(self):
        print(self.weights)

    def corrigirErro(self, entrada, erro, learning_rate):
        # como corrigidir cada elemento certo?
        self.weights[1:] += self.deltaPesoErro(1,
                                               self.no_erro[1:], learning_rate)
        self.weights[0] += self.deltaPesoErro(1,
                                              self.no_erro[0], learning_rate)

    def guardarErro(self, erro, learning_rate):
        # como pegar o input?
        # como corrigidir cada elemento certo?
        self.no_erro[1:] += self.deltaPesoErro(1, erro, learning_rate)
        #self.no_erro[0] += self.deltaPesoErro(1, erro, learning_rate)

    def deltaPesoErro(self, x, erro, learning_rate):
        # func que calcula o delta para um peso
        # Calcula para um peso
        # como calcular para uma camada toda?
        deltaPeso = learning_rate * erro * self.derivadaFunc(x) * x
        return deltaPeso

    def calcularErroBackProgation(self):
        print('implementado')


class MLP(object):
    def __init__(self, hidden_layer=1, no_neuroios_hidden=2, no_entradas=2, no_saidas=2, threshold=10, learning_rate=0.01, erro_delta_minimo=0.001):
        self.hidden_layer = hidden_layer
        self.no_neuroios_hidden = no_neuroios_hidden
        self.no_entradas = no_entradas
        self.no_saidas = no_saidas
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.erro_delta_minimo = erro_delta_minimo
        self.listaCamadas = []
        for i in range(hidden_layer):
            n_entradas = 0
            if i == 0:
                n_entradas = no_entradas
            else:
                n_entradas = no_neuroios_hidden
            m = MLPCamada(no_neuronios=self.no_neuroios_hidden,
                          no_entradas=n_entradas)
            self.listaCamadas.append(m)
        # cria a camada final
        camfinal = MLPCamada(no_neuronios=self.no_saidas,
                             no_entradas=no_neuroios_hidden)
        self.listaCamadas.append(camfinal)

    def printPesosTodos(self):
        for camada, i in zip(self.listaCamadas, range(len(self.listaCamadas))):
            print("camada ", i)
            camada.printPesos()

    def predict(self, X):
        entrada = X
        for camada in self.listaCamadas:
            resultado = camada.predict(entrada)
            entrada = resultado
        return entrada

    def corrigiErroBackPropagation(self, erro):
        # corrigir o erro da ultima
            # como ver o input da camada

        # para as outras camadas tenho que calcular o erro back propagado
            # como calcular o input
        x = 0
        erroBack = []
        camadaFrente = {}
        for camada in reversed(self.listaCamadas):
            if(x == 0):
                # ultima camada
                # erro ja calculado
                print('ultima camada')
                x = 1
                camadaFrente = camada
                camada.guardarErro(erro, self.learning_rate)

            else:
                print('nao eh ultima camada')
                # primeira coisa eh calcular o erro back progradado
                erroCamadaFrente = camadaFrente.calcularErroBackProgation()
                camada.guardarErro(erroCamadaFrente, self.learning_rate)
                camadaFrente = camada

    def train(self, X, y):
        erroAnterior = 0
        for n in range(self.threshold):
            print("Treianmento ", n)
            erroMedio = 0
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                erro = 0.5 * ((label - prediction)**2)
                print(erro)
                erroMedio += erro/(len(y))

            self.corrigiErroBackPropagation(erro)
            # self.printPesosTodos()

# %%
class MyMLP(object):
    def __init__(self, no_of_inputs, no_camadas=2, no_saidas=1, threshold=100, learning_rate=0.01, erro_delta_minimo=0.001):
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

# In[]
#execicio 1
def testarMLP(dadosX,dadosY,mlp):
    mlp.fit(dadosX,dadosY)
    print('resultados do treinamento')
    print('numerod de epocas ', mlp.n_iter_)
    return mlp

mlp = MLPClassifier(hidden_layer_sizes=(2),activation="logistic",solver="sgd",batch_size='1',verbose=True)
dadosX = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
dadosY = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])


print('execiocio 1')
print('batch vs padrao')
print('---------treinamento por padrao---------')
mlp = MLPClassifier(max_iter=2000,tol=0.0001, hidden_layer_sizes=(2),activation="logistic",solver="sgd",batch_size=1)
testarMLP(dadosX,dadosY,mlp)

print('---------treinamento por btrach---------')
mlp = MLPClassifier(max_iter=2000,hidden_layer_sizes=(2),tol=0.0001,activation="logistic",solver="sgd")
testarMLP(dadosX,dadosY,mlp)


#In[]
print('execiocio 2')
print('taxa')

taxa = 0.001
soma = 0.100
import pandas as pd
resultados = []


while taxa < 1.0:
    print(taxa)
    print('---------treinamento por padrao---------')
    mlp = MLPClassifier(learning_rate_init=taxa, max_iter=2000,tol=0.0001, hidden_layer_sizes=(2),activation="logistic",solver="sgd",batch_size=1)
    testarMLP(dadosX,dadosY,mlp)
    res = [taxa,mlp.n_iter_,'padrão']
    resultados.append(res)
    print('---------treinamento por btrach---------')
    mlp = MLPClassifier(learning_rate_init=taxa,max_iter=2000,hidden_layer_sizes=(2),tol=0.0001,activation="logistic",solver="sgd")
    testarMLP(dadosX,dadosY,mlp)
    taxa += soma
    res = [taxa,mlp.n_iter_,'batch']
    resultados.append(res)

dados = pd.DataFrame(resultados, columns=['taxa','epoch','tipo'])
dados.to_csv('resultadosEx2.csv')


# In[]
print('execiocio 3')
print('neuronios')

taxa = 0.01
soma = 1
import pandas as pd
resultados = []
n = 2


while n <= 20:
    print('numero de n ', n)
    print('---------treinamento por padrao---------')
    mlp = MLPClassifier(learning_rate_init=taxa, max_iter=2000,tol=0.0001, hidden_layer_sizes=(n),activation="logistic",solver="sgd",batch_size=1)
    testarMLP(dadosX,dadosY,mlp)
    res = [n,mlp.n_iter_,'padrão']
    resultados.append(res)
    print('---------treinamento por btrach---------')
    mlp = MLPClassifier(learning_rate_init=taxa,max_iter=2000,hidden_layer_sizes=(n),tol=0.0001,activation="logistic",solver="sgd")
    testarMLP(dadosX,dadosY,mlp)
    n += soma
    res = [n,mlp.n_iter_,'batch']
    resultados.append(res)

dados = pd.DataFrame(resultados, columns=['n','epoch','tipo'])
dados.to_csv('resultadosEx2.csv')


# In[]


print("exercicio 2")
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
iris = datasets.load_iris()

mlp = MLPClassifier(hidden_layer_sizes=(20,))

scores = cross_val_score(mlp, iris.data, iris.target, cv=5, scoring='f1_macro')
print(scores)

# In[]
print('exercicio 3')
import pandas as pd 
data = pd.read_csv("dataset_Facebook.csv") 
# Preview the first 5 lines of the loaded data 
data.head()
data2 = pd.get_dummies(data)
data2.head()
data2.fillna(0, inplace=True)

mlp = MLPRegressor(hidden_layer_sizes=(20,))
dadosY = data2['total']
dadosX = data2.drop(columns='total')
print(dadosX.head())
print(dadosY.head())

scores = cross_val_score(mlp, dadosX, dadosY, cv=5, scoring='f1_macro')
print(scores)



# transformar os dados

#%%

#%%
