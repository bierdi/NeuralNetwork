package testapplication;

import activationfunctions.ActivationFunction;
import activationfunctions.Identity;
import activationfunctions.ReLU;
import initfunctions.HeInit;
import initfunctions.InitFunction;
import java.io.FileNotFoundException;
import lossfunctions.SquaredDeviation;
import neuralnetwork.mlp.MLP;
import optimizers.GradientDescent;
import optimizers.Optimizer;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class TestApp {
    
    public static void main(String[] args) throws FileNotFoundException {
        
        /*
        MLP mlp = new MLP(new MLPLayer[] {new MLPLayer(new Matrix(new float[][] {{1},{2},{3}}), new Vector(new float[] {1,2,3}), new ReLU()), 
            new MLPLayer(new Matrix(new float[][] {{1,2,3}}), new Vector(new float[] {3}), new ReLU())}, 
            new SquaredDeviation());
        Vector result = mlp.forwardPass(new Vector(new float[] {2}));
        
        System.out.println(result);
        */
        
        Optimizer gradientDescent = new GradientDescent(0.005f, 1);
        
        Function sinus = (float f) -> (float) Math.sin(f * 10);
        
        MLP sinusMLP = new MLP(new int[] {30, 30, 30, 1}, 1, new ActivationFunction[] {new ReLU(), new ReLU(), new ReLU(), new Identity()}, new InitFunction[] {new HeInit(1234), new HeInit(1234), new HeInit(1234), new HeInit(123)}, new SquaredDeviation());
        
        int learningSteps = 50000;
        
        for (int i = 0; i < learningSteps; i++) {
            float f = (float) Math.random() * 2 - 1;
            sinusMLP.backwardPass(new Vector(new float[] {f}), new Vector(new float[] {sinus.apply(f)}));
            sinusMLP.learn(gradientDescent);
        }

        System.out.println(sinus.apply(0.6f));
        System.out.println(sinusMLP.forwardPass(new Vector(new float[] {0.6f})));              

    }
    
}
