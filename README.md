# NeuralNetwork

Ein MLP und ein CNN von Grund auf in Java implementiert. Aktuell steht eine Demo zur Verfügung, die mithilfe des MLPs eine Sinuskurve approximiert.

## Status

| Funktion | Status |
|---|---|
| Tensoren (Vektoren, Matrizen, Views auf Tensoren) | ✅ Fertig |
| Multilayer perceptron (MLP) | ✅ Fertig |
| Backpropagation | ✅ Fertig |
| Aktivierungsfunktionen: ReLU, Identity, Sigmoid | ✅ Fertig |
| Loss: Quadratische Abweichung | ✅ Fertig |
| Optimizers: Gradientenabstieg mit Momentum und Learning Rate Decay | ✅ Fertig |
| Initialisierung: Gleichverteilung, Xavier, HeInit | ✅ Fertig |
| Softmax + cross-entropy | ⚙️ in Arbeit |
| Convolutional neural network (CNN) | ⚙️ in Arbeit |
| GUI: Ziffern zeichnen und klassifizieren | ⚙️ in Arbeit |

## Demo

`TestApp` trainiert ein MLP (1 → 30 → 30 → 30 → 1, ReLU) auf eine Sinus Funktion im Intervall [0;10] (gestaucht auf [-1;1])

Zufallsgenerator ist nicht geseedet.

```
Beispielausgabe:



```

## Ausführen

Braucht mindestens Java 11

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
