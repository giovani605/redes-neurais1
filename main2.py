# In[]
import numpy as np  # Biblioteca de manipulacao de arrays Numpy
from sklearn.neural_network import MLPClassifier

class MLPCamada(object):
    def __init__(self,no_neuronios=2,no_entradas=2):
        self.weights = np.random.rand(no_entradas + 1, no_neuronios)
        self.no_neuronios = no_neuronios
        self.no_entradas = no_entradas
    
    
    def predictFunc(self, x):
        return (1 / (1 + (pow(np.e,-x))))
    
    def derivadaFunc(self, x):
        return self.predictFunc(x)*(1 - self.predictFunc(x))

    def predict(self, inputs):
        # multipla as entradas pelo pesos
        lista = []
        for i in range(self.no_neuronios):
            neuron = self.weights[:,i]
            res = self.predictFunc(np.sum(np.dot(inputs, neuron[1:]) + neuron[0])) 
            lista.append(res)
        
        return np.array(lista)

    def printPesos(self):
        print(self.weights)

    def corrigirErro(self,erro,learning_rate):
        # como pegar o input?
        # como corrigidir cada elemento certo?
        self.weights[1:] += self.deltaPesoErro(1,erro,learning_rate)
        self.weights[0] += self.deltaPesoErro(1,erro,learning_rate)

    def deltaPesoErro(self, x, erro, learning_rate):
        # func que calcula o delta para um peso
        # Calcula para um peso
        # como calcular para uma camada toda?
        deltaPeso = learning_rate * erro * self.derivadaFunc(x) * x
        return deltaPeso
    
    



class MLP(object):
    def __init__(self,hidden_layer=1,no_neuroios_hidden=2,no_entradas=2,no_saidas=2
    ,threshold=10, learning_rate=0.01, erro_delta_minimo=0.001):
        self.hidden_layer = hidden_layer
        self.no_neuroios_hidden = no_neuroios_hidden
        self.no_entradas = no_entradas
        self.no_saidas = no_saidas
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.erro_delta_minimo=erro_delta_minimo
        self.listaCamadas = []
        for i in range(hidden_layer):
            n_entradas = 0
            if i == 0:
                n_entradas = no_entradas
            else:
                n_entradas = no_neuroios_hidden
            m = MLPCamada(no_neuronios=self.no_neuroios_hidden,no_entradas=n_entradas)
            self.listaCamadas.append(m)
        # cria a camada final
        camfinal = MLPCamada(no_neuronios=self.no_saidas,no_entradas=no_neuroios_hidden)
        self.listaCamadas.append(camfinal)
    
    def printPesosTodos(self):
        for camada,i in zip(self.listaCamadas,range(len(self.listaCamadas))):
            print("camada ",i)
            camada.printPesos()


    def predict(self,X):
        entrada = X
        for camada in self.listaCamadas:
            resultado = camada.predict(entrada)
            entrada = resultado
        return entrada

    def corrigiErroBackPropagation(self,erro):
        # corrigir o erro da ultima
            # como ver o input da camada


        # para as outras camadas tenho que calcular o erro back propagado        
            # como calcular o input
        for camada in reversed(self.listaCamadas):
            camada.corrigirErro(erro,self.learning_rate)
            

    def train(self,X,y):
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
            #self.printPesosTodos()
            

# In[]
print("exercicio 1")
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

dadosX = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
dadosY = np.array([0, 1, 1, 0])
mlp.fit(dadosX,dadosY)
mlp.predict(dadosX)


# In[]
print("exercicio 2")

# In[]
print("exercicio 3")


# In[]
'''
print("Erro Medio ", erroMedio)
            print("Variacao do erro:")
            print((erroMedio - erroAnterior))
            if abs((erroMedio - erroAnterior)) <= self.erro_delta_minimo:
                break
            else:
                erroAnterior = erroMedio

'''




#%%
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


