# NeuralNetwork

Ein MLP und ein CNN von Grund auf in Java implementiert. Aktuell steht eine Demo zur Verfügung, die das MLP nutzt um Ziffern zu klassifizieren. Trainiert wurde auf dem MNIST Datensatz der unter [MNIST](https://github.com/phoebetronic/mnist) zur Verfügung steht. 

## Status

| Funktion | Status |
|---|---|
| Tensoren (Vektoren, Matrizen, Views auf Tensoren) | Fertig |
| Multilayer perceptron (MLP) | Fertig |
| Backpropagation | Fertig |
| Aktivierungsfunktionen: ReLU, Identity, Sigmoid | Fertig |
| Loss: Quadratische Abweichung | Fertig |
| Optimizers: Gradientenabstieg mit Momentum und Learning Rate Decay | Fertig |
| Initialisierung: Gleichverteilung, Xavier, HeInit | Fertig |
| Softmax + cross-entropy | Fertig |
| Convolutional neural network (CNN) | in Arbeit |
| GUI: Ziffern zeichnen | in Arbeit |

## Demo

`TestApp` trainiert ein MLP (784 → 784 → 10, ReLU, Softmax, Cross Entropy) auf MNIST

Erreicht wird eine Genauigkeit von ca. 97%. Während des Trainings und der Evaluierung wird der Fortschritt laufend auf der Konsole ausgegeben. 

## Ausführen

Braucht mindestens Java 26

```bash
java -jar NeuralNetwork.jar
```

## Struktur

```
src/
├── tensors/              Tensor, Vector, Matrix
├── neuralnetwork/        Network und Layer
│   ├── mlp/              MLP
│   └── cnn/              CNN (in Arbeit)
├── activationfunctions/
├── lossfunctions/
├── optimizers/
├── initfunctions/
├── exceptions/
└── testapplication/      Demo
```
