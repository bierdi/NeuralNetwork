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
import tensors.Matrix;
import tensors.Tensor;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class TestApp {
    
    public static void main(String[] args) throws FileNotFoundException {
        
        //testTensors();
        
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
    
    private static void testTensors() {
        
        Vector v0 = new Vector(new float[] {1,2,3});
        Vector v1 = new Vector(new float[] {4,5,6});
        v0.addTensor(v1);
        
        System.out.println(v0);
        
        Matrix m = new Matrix(new float[][] {{1,2,3},{1,2,3},{1,2,3}});
        
        System.out.println(m);
        
        Vector result = Matrix.multiplyMatrixVector(m, v0);
        
        System.out.println(result);
        
        System.out.println(""+m.getValue(1,1));
        
        m = new Matrix(new float[][] {{1,2,3},{1,2,3},{1,2,3}});
        Matrix m1 = new Matrix(new float[][] {{2,3,4},{2,3,4},{2,3,4}});
        
        Tensor t = Tensor.addTensors(m, m1);
        
        System.out.println(t);
        
        System.out.println(m);
        Matrix test = m.deepCopy();
        System.out.println(test);
        
        System.out.println("========Ultimate Test========");
        System.out.println(m);
        System.out.println(m1);
        
        Matrix testResult = Matrix.multiplyMatrizes(m,m1);
        
        System.out.println(testResult);
        
        System.out.println(testResult.getValue(1,1));
        
    }
    
}
