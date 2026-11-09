package neuralnetwork.mlp;

import activationfunctions.ActivationFunction;
import exceptions.DimensionMismatchException;
import initfunctions.InitFunction;
import java.util.Scanner;
import neuralnetwork.Layer;
import tensors.Matrix;
import tensors.Tensor;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class MLPLayer extends Layer {
    
    public MLPLayer(Matrix weights, Vector biases, ActivationFunction activator) throws DimensionMismatchException {
        
        if (weights == null || biases == null || activator == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (weights.getHeight() != biases.getSize()) {
            throw new DimensionMismatchException("Dimensions do not fit");
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
    public Vector forwardPass(Tensor input, boolean dropoutActive) throws DimensionMismatchException {
        
        inputCache = input;
        Tensor tmpResult = Vector.addVectors(Matrix.multiplyMatrixVector(Matrix.tensorToMatrix(weights), Vector.tensorToVector(input)), Vector.tensorToVector(biases));
        tmpResultCache = tmpResult.deepCopy();
        activator.apply(tmpResult);
        
        return Vector.tensorToVector(tmpResult);
    }

    @Override
    public Vector backwardPass(Tensor gradient, Vector globalGradient) throws DimensionMismatchException {
        
        Tensor tmpResultCacheCopy = tmpResultCache.deepCopy();
        
        activator.derivative(tmpResultCacheCopy);
        gradient.multiplyTensorValues(tmpResultCacheCopy);
        
        Matrix weightDerivatives = Matrix.createFromVectors(Vector.tensorToVector(inputCache), Vector.tensorToVector(gradient));
        
        Matrix weightGradient = Matrix.tensorToMatrix(Tensor.createView(weights.getDimensions(), globalGradient, new int[] {1,1}, new int[] {gradientIndex}));
        weightGradient.addMatrix(weightDerivatives);
        
        Vector biasGradient = Vector.tensorToVector(Tensor.createView(biases.getDimensions(), globalGradient, new int[] {1}, new int[] {gradientIndex + weights.getSize()}));
        biasGradient.addVector(Vector.tensorToVector(gradient));
        
        return Matrix.multiplyMatrixVector(Matrix.transposeMatrix(Matrix.tensorToMatrix(weights)), Vector.tensorToVector(gradient));
    }

    @Override
    public void learn(Vector globalGradient) throws DimensionMismatchException {
        
        Matrix weightGradient = Matrix.tensorToMatrix(Tensor.createView(weights.getDimensions(), globalGradient, new int[] {1,1}, new int[] {gradientIndex}));
        Vector biasGradient = Vector.tensorToVector(Tensor.createView(biases.getDimensions(), globalGradient, new int[] {1}, new int[] {gradientIndex + weights.getSize()}));
        
        Matrix.tensorToMatrix(weights).addMatrix(weightGradient);
        Vector.tensorToVector(biases).addTensor(biasGradient);
        
    }

    @Override
    public int[] getInputDimensions() {
        return new int[] {Matrix.tensorToMatrix(weights).getWidth()};
    }

    @Override
    public int[] getOutputDimensions() {
        return new int[] {biases.getSize()};
    }

    @Override
    public int getNumberOfParameters() {
        return weights.getSize() + biases.getSize();
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
    public MLPLayer deepCopy() throws DimensionMismatchException {
        return new MLPLayer(Matrix.tensorToMatrix(weights), Vector.tensorToVector(biases), activator);
    }

    @Override
    public void appendToStringBuilder(StringBuilder sb) {
        
        sb.append(this.getLayerName());
        sb.append("\n");
        sb.append(this.getActivationFunctionName());
        sb.append("\n");
        Matrix w = Matrix.tensorToMatrix(weights);
        sb.append(w.getWidth());
        sb.append("\n");
        sb.append(w.getHeight());
        sb.append("\n");

        for (int j = 0; j < this.getNumberOfParameters(); j++) {
            sb.append(this.getParameterAt(j));
            sb.append(" ");
        }
            
    }

    @Override
    protected void buildFromFile(Scanner scanner) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
}
