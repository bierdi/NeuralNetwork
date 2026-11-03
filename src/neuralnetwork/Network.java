package neuralnetwork;

import matrixvector.Vector;
import lossfunctions.SquaredDeviation;
import matrixvector.Matrix;
import activationfunctions.ActivationFunction;
import lossfunctions.LossFunction;

/**
 *
 * @author bierdi
 */
public class Network {
    
    //Adam
    
    private static final double EPSILON = 1E-8;
    private static final double BETA_ONE = 0.9;
    private static final double BETA_TWO = 0.999;
    
    private Vector m;
    private Vector v;
    private int time = 1;
    
    //Adam
    
    private final int numLayers;
    private final Layer[] layers;
    
    private Vector contributions;
    
    private final LossFunction lossFunction = new SquaredDeviation();
    
    //create Network with random parameters and given number of layers and neurons
    public Network(int[] numNeurons, ActivationFunction[] activators) {

        numLayers = numNeurons.length;
        layers = new Layer[numNeurons.length];
        
        if (numLayers != activators.length) {
            throw new IllegalArgumentException("Every layer needs exactly one activator");
        }
        
        layers[0] = new Layer(numNeurons[0], numNeurons[0], activators[0]);
        
        for (int i = 1; i < numLayers; i++) {
            layers[i] = new Layer(numNeurons[i], numNeurons[i-1], activators[i]);
        }
        
        initContributions();
        
    }
    
    //create Network from given parameters
    public Network(Vector[] biases, Matrix[] weights, int[] numNeurons, ActivationFunction[] activators) {
        
        numLayers = numNeurons.length;
        layers = new Layer[numNeurons.length];    
        
        if (numLayers != activators.length) {
            throw new IllegalArgumentException("Every layer needs exactly one activator");
        }
        
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new Layer(biases[i], weights[i], numNeurons[i], activators[i]);
        }
        
        initContributions();
        
    }
    
    private void initContributions() {
       
        int numParams = layers[0].getNumNeurons() + layers[0].getNumNeurons() * layers[0].getNumNeurons();
        
        for (int i = 1; i < numLayers; i++) {
            numParams += layers[i].getNumNeurons() + layers[i-1].getOutputs().getSize() * layers[i].getNumNeurons();
        }
        
        contributions = new Vector(numParams);
        m = new Vector(numParams);
        v = new Vector(numParams);
        
    }
    
    public Vector forwardPass(Vector input) {
        
        Vector tmp = input;
        
        for (int i = 0; i < numLayers; i++) {
            tmp = layers[i].calculateOutputs(tmp);
        }
        
        return tmp;
    }
    
    private void backwardPass(Matrix gradient, Vector inputs) {
        
        int position = 0;
        
        for (int i = numLayers - 1; i > 0; i--) {
            layers[i].deriveActivation(gradient);
            layers[i].deriveBiases(contributions, position, gradient);
            layers[i].deriveWeights(contributions, layers[i-1].getOutputs(), position, gradient);
            gradient = layers[i].deriveInputs(gradient);
            
            position += layers[i].getNumNeurons() + layers[i].getNumNeurons() * layers[i-1].getOutputs().getSize();
        }
        
        layers[0].deriveActivation(gradient);
        layers[0].deriveBiases(contributions, position, gradient);
        layers[0].deriveWeights(contributions, inputs, position, gradient);
        
    }
    
    public double learn(Vector inputs, Vector expectation, double rate) {
        
        Vector output = forwardPass(inputs);
        double loss = lossFunction.apply(output, expectation);
        Matrix lossDerivative = lossFunction.applyDerivative(output, expectation);
        backwardPass(lossDerivative, inputs);
        
        m.multiplyScalar(BETA_ONE);
        m.addVector(Vector.multiplyScalar(contributions, 1 - BETA_ONE));
        v.multiplyScalar(BETA_TWO);
        v.addVector(Vector.multiplyScalar(Vector.squareComponents(contributions), 1 - BETA_TWO));
        
        Vector tmpM = Vector.multiplyScalar(m, 1 / (1 - Math.pow(BETA_ONE, time)));
        Vector tmpV = Vector.multiplyScalar(v, 1 / (1 - Math.pow(BETA_TWO, time)));
        
        tmpM.multiplyScalar(-rate);
        tmpV.squareRootComponents();
        tmpV.addScalar(EPSILON);
        tmpV.invertComponents();
        Vector update = Vector.multiplyComponents(tmpM, tmpV);
        
        int position = 0;
        
        for (int i = numLayers - 1; i > 0; i--) {
            layers[i].updateParameters(update, position);
            position += layers[i].getNumNeurons() + layers[i].getNumNeurons() * layers[i-1].getOutputs().getSize();
        }
        
        layers[0].updateParameters(update, position);
        
        contributions.multiplyScalar(0);
        time++;
        
        return loss;
    }
    
    //Learning with batches
    public double learn(Vector[] inputs, Vector[] expectations, double rate) {
        
        if (inputs == null || expectations == null) {
            throw new IllegalArgumentException("Inputs and expectations must not be null");
        }
        
        if (inputs.length == 0 || expectations.length == 0) {
            throw new IllegalArgumentException("Inputs and expectations must not be empty");
        }
        
        if (inputs.length != expectations.length) {
            throw new IllegalArgumentException("Number of inputs and expectations not matching");
        }
        
        double loss = 0;
        
        for (int i = 0; i < inputs.length; i++) {
            Vector output = forwardPass(inputs[i]);
            loss += lossFunction.apply(output, expectations[i]);
            Matrix lossDerivative = lossFunction.applyDerivative(output, expectations[i]);
            backwardPass(lossDerivative, inputs[i]);
        }
        
        contributions.multiplyScalar(1.0 / inputs.length);
        loss /= inputs.length;
        
        m.multiplyScalar(BETA_ONE);
        m.addVector(Vector.multiplyScalar(contributions, 1-BETA_ONE));
        v.multiplyScalar(BETA_TWO);
        v.addVector(Vector.multiplyScalar(Vector.squareComponents(contributions), 1 - BETA_TWO));
        
        Vector tmpM = Vector.multiplyScalar(m, 1 / (1 - Math.pow(BETA_ONE, time)));
        Vector tmpV = Vector.multiplyScalar(v, 1 / (1 - Math.pow(BETA_TWO, time)));
        
        tmpM.multiplyScalar(-rate);
        tmpV.squareRootComponents();
        tmpV.addScalar(EPSILON);
        tmpV.invertComponents();
        Vector update = Vector.multiplyComponents(tmpM, tmpV);
        
        int position = 0;
        
        for (int i = numLayers - 1; i > 0; i--) {
            layers[i].updateParameters(update, position);
            position += layers[i].getNumNeurons() + layers[i].getNumNeurons() * layers[i-1].getOutputs().getSize();
        }
        
        layers[0].updateParameters(update, position);
        
        contributions.multiplyScalar(0);
        time++;
        
        return loss;
    }
    
    public int getNumLayers() {
        return numLayers;
    }
    
}
