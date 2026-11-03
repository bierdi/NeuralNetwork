package testapplication;

import activationfunctions.Relu;
import matrixvector.Matrix;
import matrixvector.Vector;
import activationfunctions.ActivationFunction;
import activationfunctions.Identity;
import java.util.Random;
import neuralnetwork.Network;

/**
 *
 * @author bierdi
 */
public class TestApp {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        Random r = new Random(100200300);
        
        ActivationFunction[] activators = new ActivationFunction[]{new Identity(), new Relu(), new Identity()};
        Network network = new Network(new int[] {1, 30, 1}, activators);
        
        Function sinus = (double d) -> Math.sin(d);
        
        for (int i = 0; i < 1000; i++) {
            
            Vector[] inputs = new Vector[50];
            Vector[] expectations = new Vector[50];
            
            for (int j = 0; j < 50; j++) {
                double x = r.nextDouble()*2-1;
                inputs[j] = new Vector(new double[] {x});
                expectations[j] = new Vector(new double[] {sinus.apply(x)});
            }
        
            network.learn(inputs, expectations, 0.01);
        }
        
        Vector result = network.forwardPass(new Vector(new double[] {1}));
        
        System.out.println("Expected: " + sinus.apply(1));
        System.out.println("After training: "+result);
        
    }
    
    private static void testy() {
        long start = System.currentTimeMillis();
        
        Vector[] biases = {new Vector(1), new Vector(1), new Vector(new double[] {1})};
        Matrix[] weights = 
            {new Matrix(new double[][] {{1}}),
             new Matrix(new double[][] {{2}}),
             new Matrix(new double[][] {{3}})};
        
        ActivationFunction[] activators = {new Relu(), new Relu(), new Relu()};
        
        Network network = new Network(biases, weights, new int[] {1,1,1}, activators);
        Vector input = new Vector(1);
        input.setValue(0, 1);
        Vector output = network.forwardPass(input);
        
        network.learn(input, new Vector(new double[]{6}), 0.01);
        
        Function linear = (double d) -> 4*d+3;
        
        activators = new ActivationFunction[]{new Identity()};
        Network linearRegression = new Network(new int[]{1}, activators);
        Vector result;
        
        for (int i = 0; i < 100; i++) {
            double x = Math.random()*2-1;
        
            linearRegression.learn(new Vector(new double[] {x}), new Vector(new double[] {linear.apply(x)}), 0.1);
        }
        
        result = linearRegression.forwardPass(new Vector(new double[] {50}));
        
        long duration = System.currentTimeMillis() - start;
        
        System.out.println("Time spent: " + duration + "ms");
        
        System.out.println("Output: " + output);
        
        System.out.println("Expected: " + linear.apply(50));
        System.out.println("After training: " + result);
    }
    
}
