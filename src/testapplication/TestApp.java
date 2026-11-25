package testapplication;

import activationfunctions.ActivationFunction;
import activationfunctions.ReLU;
import activationfunctions.SoftMax;
import exceptions.DimensionMismatchException;
import initfunctions.HeInit;
import initfunctions.InitFunction;
import java.io.File;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;
import lossfunctions.CrossEntropy;
import neuralnetwork.mlp.MLP;
import optimizers.GradientDescentMomentum;
import optimizers.Optimizer;
import tensors.Vector;
import testapplication.mnist.LabelImagePair;
import testapplication.mnist.MNISTLoader;

/**
 *
 * @author bierdi
 */
public class TestApp {
    
    public static void main(String[] args) throws DimensionMismatchException, FileNotFoundException, IOException, ClassNotFoundException, NoSuchMethodException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
        
        Optimizer optimizer = new GradientDescentMomentum(0.0008f, 1, 0.9f, 0.0f, 0);
        
        MLP digitClassifier = new MLP(
            new int[] {784, 784, 10},
            784,
            new ActivationFunction[] {new ReLU(), new ReLU(), new SoftMax()},
            new InitFunction[] {new HeInit(0), new HeInit(1), new HeInit(2)},
            new CrossEntropy(),
            0.0f,
            new float[] {0.0f, 0.0f, 0.0f});
        
        List<LabelImagePair> trainingData = MNISTLoader.getData(new File("data/training/mnist_train.csv"));
        System.out.println("Loaded training data");
        
        int count = 0;
        
        for (int i = 0; i < 10; i++) {
        
            for (LabelImagePair labelImagePair : trainingData) {

                count++;

                Vector label = new Vector(10);
                label.setValue(1, labelImagePair.label());
                
                digitClassifier.backwardPass(labelImagePair.image(), label);
                digitClassifier.learn(optimizer);

                if (count % 1000 == 0) {
                    System.out.println(count);
                }

            }
            
        }
        
        digitClassifier.saveToFile(new File("data/digitClassifier"));
        System.out.println("Saved");
        digitClassifier = new MLP(new File("data/digitClassifier"));
        
        List<LabelImagePair> testData = MNISTLoader.getData(new File("data/training/mnist_test.csv"));
        
        int hits = 0;
        count = 0;
        
        for (LabelImagePair labelImagePair : testData) {
            
            count++;
            
            Vector output = digitClassifier.forwardPass(labelImagePair.image());
            
            int max = 0;
            float maxValue = -1.0f;
            
            for (int i = 0; i < output.getSize(); i++) {
                if (output.getValue(i) > maxValue) {
                    maxValue = output.getValue(i);
                    max = i;
                }
            }
            
            if (max == labelImagePair.label()) {
                hits++;
            }
            
            if (count % 1000 == 0) {
                System.out.println(count);
            }
            
        }
        
        System.out.println("Accuracy: " + (double) hits / testData.size());
        
    }
    
}
