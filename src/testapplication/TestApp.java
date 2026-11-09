package testapplication;

import activationfunctions.ActivationFunction;
import activationfunctions.Identity;
import activationfunctions.ReLU;
import exceptions.DimensionMismatchException;
import initfunctions.HeInit;
import initfunctions.InitFunction;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import lossfunctions.SquaredDeviation;
import neuralnetwork.mlp.MLP;
import optimizers.GradientDescentMomentum;
import optimizers.Optimizer;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class TestApp {
    
    public static void main(String[] args) throws FileNotFoundException, DimensionMismatchException, IOException {
        
        //Gradienten Abstieg mit Trägheitsmomentum, Parameter: Lernrate, Batchgröße, Momentum Hyperparameter, Decay Rate für Learning Rate, Schrittzahl ab der Decay nicht mehr greift
        Optimizer gradientDescent = new GradientDescentMomentum(0.005f, 1, 0.9f, 0.0001f, 45000);
        
        // Sinusfunktion zum Label generieren
        Function sine = (float f) -> (float) Math.sin(f * 10);
        
        //Seed für Initialisierung der Gewichte mit HeInit
        int seed = 1234;
        
        //MLP mit 4 Schichten (30,30,30,1), Aktivierungen (ReLU, ReLU, ReLU, Identity), Lambda für L2 Regularisierung bei 0, Dropout Rate bei 0 für alle Layer
        MLP sineMLP = new MLP(new int[] {30, 30, 30, 1}, 1, new ActivationFunction[] {new ReLU(), new ReLU(), new ReLU(), new Identity()}, new InitFunction[] {new HeInit(seed), new HeInit(seed), new HeInit(seed), new HeInit(seed)}, new SquaredDeviation(), 0.0f, new float[] {0.0f,0.0f,0.0f,0.0f});
        
        //50000 Lernschritte
        int learningSteps = 50000;
        
        //Trainingsschleife
        for (int i = 0; i < learningSteps; i++) {
            float f = (float) Math.random() * 2 - 1;
            try {
                sineMLP.backwardPass(new Vector(new float[] {f}), new Vector(new float[] {sine.apply(f)}));
            } catch (DimensionMismatchException ex) {
                System.out.println(ex.getMessage());
            }
            sineMLP.learn(gradientDescent);
        }

        try {
            System.out.println("Expected for x=-0.8: " + sine.apply(-0.8f));
            System.out.println("Prediction for x=-0.8: " + sineMLP.forwardPass(new Vector(new float[] {-0.8f})));
            System.out.println("Expected for x=-0.6: " + sine.apply(-0.6f));
            System.out.println("Prediction for x=-0.6: " + sineMLP.forwardPass(new Vector(new float[] {-0.6f})));
            System.out.println("Expected for x=-0.4: " + sine.apply(-0.4f));
            System.out.println("Prediction for x=-0.4: " + sineMLP.forwardPass(new Vector(new float[] {-0.4f})));
            System.out.println("Expected for x=-0.2: " + sine.apply(-0.2f));
            System.out.println("Prediction for x=-0.2: " + sineMLP.forwardPass(new Vector(new float[] {-0.2f})));
            System.out.println("Expected for x=0: " + sine.apply(0.0f));
            System.out.println("Prediction for x=0: " + sineMLP.forwardPass(new Vector(new float[] {0.0f})));
            System.out.println("Expected for x=0.2: " + sine.apply(0.2f));
            System.out.println("Prediction for x=0.2: " + sineMLP.forwardPass(new Vector(new float[] {0.2f})));
            System.out.println("Expected for x=0.4: " + sine.apply(0.4f));
            System.out.println("Prediction for x=0.4: " + sineMLP.forwardPass(new Vector(new float[] {0.4f})));
            System.out.println("Expected for x=0.6: " + sine.apply(0.6f));
            System.out.println("Prediction for x=0.6: " + sineMLP.forwardPass(new Vector(new float[] {0.6f})));
            System.out.println("Expected for x=0.8: " + sine.apply(0.8f));
            System.out.println("Prediction for x=0.8: " + sineMLP.forwardPass(new Vector(new float[] {0.8f})));
        } catch (DimensionMismatchException ex) {
            System.out.println(ex.getMessage());
        }

    }
    
}
