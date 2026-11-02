package neuralnetwork.mlp;

import activationfunctions.ActivationFunction;
import initfunctions.InitFunction;
import neuralnetwork.Layer;
import tensors.Matrix;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class MLPLayer extends Layer<Matrix, Vector, Vector, Vector> {
    
    public MLPLayer(Matrix weights, Vector biases, ActivationFunction activator) {
        
        if (weights == null || biases == null || activator == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (weights.getHeight() != biases.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        this.weights = weights.deepCopy();
        this.biases = biases.deepCopy();
        
        this.activator = activator;
        
    }
    
    public MLPLayer(int numNeurons, int numInputs, ActivationFunction activator, InitFunction init) {
        
        if (init == null || activator == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        float[][] weightValues = new float[numNeurons][numInputs];
        
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numNeurons; j++) {
                weightValues[j][i] = init.generateValue(numInputs);
            }
        }
        
        weights = new Matrix(weightValues);
        biases = new Vector(numNeurons);
        
        this.activator = activator;
        
    }
    
    @Override
    public Vector forwardPass(Vector input) {
        
        inputCache = input;
        Vector tmpResult = Vector.addVectors(Matrix.multiplyMatrixVector(weights, input), biases);
        tmpResultCache = tmpResult.deepCopy();
        activator.apply(tmpResult);
        
        return tmpResult;
    }

    @Override
    public Vector backwardPass(Vector gradient, Vector globalGradient) {
        
        Vector tmpResultCacheCopy = tmpResultCache.deepCopy();
        
        activator.derivative(tmpResultCacheCopy);
        gradient.multiplyVector(tmpResultCacheCopy);
        
        Matrix weightDerivatives = Matrix.createFromVectors(inputCache, gradient);
        
        Matrix weightGradient = new Matrix(weights.getWidth(), weights.getHeight(), globalGradient, gradientIndex, gradientIndex + weights.getSize());       
        weightGradient.addMatrix(weightDerivatives);
        
        Vector biasGradient = new Vector(biases.getSize(), globalGradient, gradientIndex + weights.getSize(), gradientIndex + weights.getSize() + biases.getSize());
        biasGradient.addVector(gradient);
        
        return Matrix.multiplyMatrixVector(Matrix.transposeMatrix(weights), gradient);
    }

    @Override
    public void learn(Vector globalGradient) {
        
        Matrix weightGradient = new Matrix(weights.getWidth(), weights.getHeight(), globalGradient, gradientIndex, gradientIndex + weights.getSize());
        Vector biasGradient = new Vector(biases.getSize(), globalGradient, gradientIndex + weights.getSize(), gradientIndex + weights.getSize() + biases.getSize());
        
        weights.addMatrix(weightGradient);
        biases.addVector(biasGradient);
        
    }

    @Override
    public int[] getInputDimensions() {
        return new int[] {weights.getWidth()};
    }

    @Override
    public int[] getOutputDimensions() {
        return new int[] {biases.getSize()};
    }

    @Override
    public int getNumberOfParameters() {
        return weights.getHeight() * weights.getWidth() + biases.getSize();
    }

    @Override
    public float getParameterAt(int parameterIndex) {
        
        if (parameterIndex < 0 || parameterIndex >= getNumberOfParameters()) {
            throw new IllegalArgumentException("Parameter index out of bounds");
        }
        
        if (parameterIndex < weights.getSize()) {
            return weights.getValueFlat(parameterIndex);
        } else {
            return biases.getValueFlat(parameterIndex - weights.getSize());
        }
        
    }
    
    @Override
    public void setParameterAt(int parameterIndex, float valueChange) {
        
        if (parameterIndex < 0 || parameterIndex >= getNumberOfParameters()) {
            throw new IllegalArgumentException("Parameter index out of bounds");
        }
        
        if (parameterIndex < weights.getSize()) {
            weights.setValueFlat(valueChange, parameterIndex);
        } else {
            biases.setValueFlat(valueChange, parameterIndex - weights.getSize());
        }
        
    }

    @Override
    public MLPLayer deepCopy() {
        return new MLPLayer(weights, biases, activator);
    }
    
}
