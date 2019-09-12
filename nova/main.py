import random
import numpy as np
from Tarefa3 import Network

# Exercicio 3.1
n_iter_train = 100
inputs31 = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs31 = [[0, 0], [0, 1], [0, 1], [1, 1]]

network311 = Network(2, 2, learning_rate=0.1, model='perceptron')
network311.train_default(inputs31, outputs31, n_iter_train, verbose=False)
network312 = Network(2, 2, learning_rate=0.1, model='perceptron')
network312.train_batch(inputs31, outputs31, n_iter_train, verbose=False)


# Exercicio 3.2
n_iter_train = 100
inputs32 = []
outputs32 = []
for p in range(15):
    z = np.pi/7 * p
    f1 = np.sin(z)
    f2 = np.cos(z)
    f3 = z
    inputs32.append([f1, f2, f3])
    output = -np.pi + 0.565*f1+2.657*f2+0.674*f3
    outputs32.append([output])

network32 = Network(1, 3, learning_rate=0.1, model='adaline')
network32.train_default(inputs32, outputs32, n_iter_train, verbose=False)
results32 =[]
for i in range(15):
    r = network32.infer(inputs32[i])
    results32.append(r)


# # Exercicio 3.3
n_iter_train = 1000
inputs33 = inputs32
outputs33 = outputs32

network33 = Network(1, 3, learning_rate=0.075, model='adaline')
network33.train_batch(inputs33, outputs33, n_iter_train, verbose=False)
results33 = []
for i in range(15):
    r = network33.infer(inputs33[i])
    results33.append(r)


# Exercicio 3.4
n_iter_train = 100
inputs34_train = inputs33
outputs34_train = outputs33
inputs34_test = []
outputs34_test = []

for p in range(15):
    z = np.pi/7 * p + np.pi/14
    f1 = np.sin(z)
    f2 = np.cos(z)
    f3 = z
    inputs34_test.append([f1, f2, f3])
    output = -np.pi + 0.565*f1+2.657*f2+0.674*f3
    outputs34_test.append(output)

network34 = Network(1, 3, learning_rate=0.075, model='adaline')
network34.train_default(inputs34_train, outputs34_train, n_iter_train, verbose=False)
results34 = []
for i in range(15):
    r = network34.infer(inputs34_test[i])
    results34.append(r)

# Exercicio 3.5
n_iter_train = 100000
inputs35 = []
outputs35 = []
for p in range(50):
    z = np.pi/25 * p
    f1 = np.sin(z)
    f2 = np.cos(z)
    f3 = z
    inputs35.append([f1, f2, f3])
    output = -np.pi + 0.565*f1+2.657*f2+0.674*f3
    outputs35.append(output)

indexes = list(range(50))
random.shuffle(indexes)
inputs35_train = [inputs35[idx] for idx in indexes[:30]]
outputs35_train = [outputs35[idx] for idx in indexes[:30]]
inputs35_eval = [inputs35[idx] for idx in indexes[30:40]]
outputs35_eval = [outputs35[idx] for idx in indexes[30:40]]
inputs35_test = [inputs35[idx] for idx in indexes[40:]]
outputs35_test = [outputs35[idx] for idx in indexes[40:]]

network35 = Network(1, 3, learning_rate=0.075, model='adaline')
network35.train_default(inputs35_train, outputs35_train, n_iter_train, error_thresh=0.001,
                        eval_input_list=inputs35_eval, eval_output_list=outputs35_eval, verbose=False)

results35 = []
for i in range(len(inputs35_test)):
    r = network35.infer(inputs35_test[i])
    results35.append(r)




