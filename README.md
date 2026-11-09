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

`TestApp` trainiert ein MLP (1 → 30 → 30 → 30 → 1, ReLU) auf eine Sinus Funktion im Intervall [-1;1]

Zufallsgenerator ist nicht geseedet.

```
Beispielausgabe:

Expected for x=-0.8: -0.98935825
Prediction for x=-0.8: [-0.9705397]
Expected for x=-0.6: 0.2794155
Prediction for x=-0.6: [0.2615771]
Expected for x=-0.4: 0.7568025
Prediction for x=-0.4: [0.76563525]
Expected for x=-0.2: -0.9092974
Prediction for x=-0.2: [-0.9115073]
Expected for x=0: 0.0
Prediction for x=0: [0.0018287897]
Expected for x=0.2: 0.9092974
Prediction for x=0.2: [0.9104122]
Expected for x=0.4: -0.7568025
Prediction for x=0.4: [-0.7623644]
Expected for x=0.6: -0.2794155
Prediction for x=0.6: [-0.27798286]
Expected for x=0.8: 0.98935825
Prediction for x=0.8: [0.9841894]

```

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
