package neuralnetwork;

import matrixvector.Vector;
import matrixvector.Matrix;
import activationfunctions.ActivationFunction;
import initfunctions.InitFunction;
import initfunctions.RandomInit;

/**
 *
 * @author bierdi
 */
public class Layer {
    
    private static final int SEED = 1234;
    
    private final InitFunction init;
    
    private final Vector biases;
    private final Matrix weights;
    
    private final int numNeurons;
    private Vector outputs;
    
    private final ActivationFunction activator;
    
    //create Layer from given parameters
    protected Layer(Vector biases, Matrix weights, int numNeurons, ActivationFunction activator) {
        
        if (biases.getSize() != numNeurons || weights.getRows() != numNeurons) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        this.biases = biases;
        this.weights = weights;
        this.numNeurons = numNeurons;
        outputs = new Vector(numNeurons);
        
        this.activator = activator;
        
        this.init = new RandomInit(SEED);
        
    }
    
    //create Layer using random parameters
    protected Layer(int numNeurons, int numInputs, ActivationFunction activator) {
        biases = new Vector(numNeurons);
        weights = new Matrix(numInputs, numNeurons);
        this.numNeurons = numNeurons;
        outputs = new Vector(numNeurons);
        
        init = new RandomInit(SEED);
        
        generateRandomParameters();
        
        this.activator = activator;
    }
    
    protected int getNumNeurons() {
        return numNeurons;
    }
    
    private void generateRandomParameters() {
        
        for (int i = 0; i < weights.getColumns(); i++) {
            for (int j = 0; j < weights.getRows(); j++) {
                weights.setValue(i, j, init.generate());
            }
        }
        
        for (int i = 0; i < biases.getSize(); i++) {
            biases.setValue(i, init.generate());
        }
   
    }
    
    protected Vector getOutputs() {
        return outputs;
    }
    
    protected Vector calculateOutputs(Vector inputs) {
        
        outputs = Matrix.multiplyVector(weights, inputs);
        outputs.addVector(biases);
        
        for (int i = 0; i < outputs.getSize(); i++) {
            outputs.setValue(i, activator.apply(outputs.getValue(i)));
        }
        
        return outputs;
    }
    
    protected void deriveBiases(Vector contributions, int position, Matrix gradient) {
        
        for (int i = 0; i < numNeurons; i++) {
            contributions.setValue(position + i, contributions.getValue(position + i) + gradient.getValue(i, 0));
        }
        
    }
    
    protected void deriveWeights(Vector contributions, Vector inputs, int position, Matrix gradient) {
        
        position += numNeurons;
        
        Matrix derivative = Matrix.multiplyMatrix(Matrix.transposeMatrix(gradient), Vector.transposeVector(inputs));
        
        int i = 0;
        for (int j = 0; j < derivative.getColumns(); j++) {
            for (int k = 0; k < derivative.getRows(); k++) {
                contributions.setValue(position + i, contributions.getValue(position + i) + derivative.getValue(j, k));
                i++;
            }
        }
        
    }
    
    protected void deriveActivation(Matrix gradient) {
        
        for (int i = 0; i < gradient.getColumns(); i++) {
            gradient.setValue(i, 0, gradient.getValue(i,0) * activator.applyDerivative(outputs.getValue(i)));
        }
        
    }
    
    protected Matrix deriveInputs(Matrix gradient) {
        return Matrix.multiplyMatrix(gradient, weights);
    }
    
    protected void updateParameters(Vector contributions, int position) {
        
        Vector biasesContributions = new Vector(numNeurons);
        
        for (int i = 0; i < numNeurons; i++) {
            biasesContributions.setValue(i, contributions.getValue(position + i));
        }
        
        biases.addVector(biasesContributions);
        
        int k = 0;
        for (int i = 0; i < weights.getColumns(); i++) {
            for(int j = 0; j < weights.getRows(); j++) {
                weights.setValue(i, j, weights.getValue(i, j) + contributions.getValue(position + numNeurons + k));
                k++;
            }
        }
        
    }
    
}
