package testapplication;

import activationfunctions.ActivationFunction;
import activationfunctions.Identity;
import activationfunctions.ReLU;
import exceptions.DimensionMismatchException;
import initfunctions.HeInit;
import initfunctions.InitFunction;
import java.io.FileNotFoundException;
import lossfunctions.SquaredDeviation;
import neuralnetwork.mlp.MLP;
import optimizers.GradientDescent;
import optimizers.GradientDescentMomentum;
import optimizers.Optimizer;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class TestApp {
    
    public static void main(String[] args) throws FileNotFoundException, DimensionMismatchException {
        
        /*
        MLP mlp = new MLP(new MLPLayer[] {new MLPLayer(new Matrix(new float[][] {{1},{2},{3}}), new Vector(new float[] {1,2,3}), new ReLU()), 
            new MLPLayer(new Matrix(new float[][] {{1,2,3}}), new Vector(new float[] {3}), new ReLU())}, 
            new SquaredDeviation());
        Vector result = mlp.forwardPass(new Vector(new float[] {2}));
        
        System.out.println(result);
        */
        
        Optimizer gradientDescent = new GradientDescentMomentum(0.005f, 1);
        
        Function sine = (float f) -> (float) Math.sin(f * 10);
        
        int seed = 1234;
        
        MLP sineMLP = new MLP(new int[] {30, 30, 30, 1}, 1, new ActivationFunction[] {new ReLU(), new ReLU(), new ReLU(), new Identity()}, new InitFunction[] {new HeInit(seed), new HeInit(seed), new HeInit(seed), new HeInit(seed)}, new SquaredDeviation());
        
        int learningSteps = 50000;
        
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
            if (Math.abs(sine.apply(0.6f) - sineMLP.forwardPass(new Vector(new float[] {0.6f})).getValue(0)) < 0.05f) {
                System.out.println(sine.apply(0.0f));
                System.out.println(sineMLP.forwardPass(new Vector(new float[] {0.0f})));
                System.out.println(sine.apply(0.2f));
                System.out.println(sineMLP.forwardPass(new Vector(new float[] {0.2f})));
                System.out.println(sine.apply(0.4f));
                System.out.println(sineMLP.forwardPass(new Vector(new float[] {0.4f})));
                System.out.println(sine.apply(0.6f));
                System.out.println(sineMLP.forwardPass(new Vector(new float[] {0.6f})));
                System.out.println(sine.apply(0.8f));
                System.out.println(sineMLP.forwardPass(new Vector(new float[] {0.8f})));
            } else {
                System.out.println("Failure");             
            }
        } catch (DimensionMismatchException ex) {
            System.out.println(ex.getMessage());
        }

    }
    
}
