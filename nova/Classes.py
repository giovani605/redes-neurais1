import numpy as np
from Tarefa3.Functions import sat, step, linear

models = ['perceptron', 'adaline']


class Neuron:
    def __init__(self, n_inputs, weights=None, bias=None, func=sat):
        self.n_inputs = n_inputs
        if weights is None:
            self.weights = np.random.rand(n_inputs) * 2 - 1  # fica entre -1 e 1
        else:
            if len(self.weights) != n_inputs:
                raise Exception("Numero de pesos diferente do esperado")
            self.weights = weights
        if bias is None:
            self.bias = np.random.rand()
        else:
            self.bias = bias

        self.func = func

    def run(self, x):
        if len(x) != self.n_inputs:
            raise Exception("Neuronio recebendo numero de inputs (" + str(x.shape[0]) + ") diferentes do esperado ("
                            + str(self.n_inputs) + ")")
        result = self.func(x, self.weights, theta=self.bias)

        return result

    def update_weights(self, new_weights, new_bias):
        if not new_weights.shape[0] == self.n_inputs:
            raise Exception("Array de pesos com tamanho diferente do esperado")
        self.weights = np.array(new_weights)
        self.bias = new_bias


class Network:
    def __init__(self, n_neurons, n_inputs, learning_rate=0.01, model='perceptron'):
        if model not in models:
            raise Exception("Modelo de neuronio nao implementado")
        self.model = model
        self.learning_rate = learning_rate
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        if model == 'perceptron':
            func = step
        elif model == 'adaline':
            func = linear
        else:
            func = None

        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs, func=func))

    def _train_default_one_iteration(self, x, desired_output):
        x = np.array(x)
        desired_output = np.array(desired_output)

        value = self.infer(x)
        value = np.array(value)
        error = desired_output - value

        for i in range(len(self.neurons)):
            n = self.neurons[i]
            e = error[i]
            new_weights = n.weights + self.learning_rate * e * x
            new_bias = n.bias + self.learning_rate * e
            n.update_weights(new_weights, new_bias)

        return value, error

    def _train_batch_one_iteration(self, input_list, desired_output_list):
        values = []
        errors = []
        for i in range(len(input_list)):
            x = np.array(input_list[i])
            desired_output = np.array(desired_output_list[i])

            value = self.infer(x)
            value = np.array(value)
            error = desired_output - value

            values.append(value)
            errors.append(error)

        for i in range(self.n_neurons):
            n = self.neurons[i]
            delta = 0
            delta_bias = 0
            for j in range(len(input_list)):
                delta += self.learning_rate * errors[j][i] * np.array(input_list[j])
                delta_bias += self.learning_rate * errors[j][i]

            delta /= len(input_list)

            new_weights = n.weights + delta
            new_bias = n.bias + delta_bias
            n.update_weights(new_weights, new_bias)

        return values, errors

    def train_default(self, input_list, output_list, n_iter, error_thresh=None, eval_input_list=None,
                      eval_output_list=None, verbose=True):
        for i in range(n_iter):
            errors = []
            for j in range(len(input_list)):
                prediction, error = self._train_default_one_iteration(input_list[j], output_list[j])
                errors.append(error)
                if verbose:
                    print('Iteration ' + str(i) + ':\t Input: ' + str(input_list[j]) + '\t Output: ' +
                          str(prediction) + '\t Desired output: ' + str(output_list[j]), '\t error = ' + str(error))

            if error_thresh is not None:
                if eval_input_list is not None:
                    for j in range(len(eval_input_list)):
                        value = self.infer(eval_input_list[j])
                        error = eval_output_list[j] - value
                        errors.append(error)

                error = np.abs(np.array(errors)).sum()
                if error <= error_thresh:
                    if verbose:
                        print("Finished training on iteration " + str(i))
                    return

    def train_batch(self, input_list, output_list, n_iter, error_thresh=None, eval_input_list=None,
                    eval_output_list=None, verbose=True):
        for i in range(n_iter):
            predictions, errors = self._train_batch_one_iteration(input_list, output_list)
            if verbose:
                for j in range(len(predictions)):
                    print('Iteration ' + str(i) + ':\t Input: ' + str(input_list[j]) + '\t Output: ' +
                          str(predictions[j]) + '\t Desired output: ' + str(output_list[j]), '\t error = '
                          + str(errors[j]))

            if error_thresh is not None:
                if eval_input_list is not None:
                    for j in range(len(eval_input_list)):
                        value = self.infer(eval_input_list[j])
                        error = eval_output_list[j] - value
                        errors.append(error)

                error = np.abs(np.array(errors)).sum()
                if error <= error_thresh:
                    if verbose:
                        print("Finished training on iteration " + str(i))
                    return

    def infer(self, x):
        value = []
        for neuron in self.neurons:
            value.append(neuron.run(x))
        return value
