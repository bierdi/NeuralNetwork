package neuralnetwork;

import exceptions.DimensionMismatchException;
import java.io.File;
import java.io.IOException;
import lossfunctions.*;
import optimizers.Optimizer;
import tensors.Tensor;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public abstract class Network {
    
    protected Layer[] layers;
    
    protected Vector gradient; //holds partial derivatives of all parameters
    protected int[] gradientLayerMapping; //maps entry in gradient to corresponding layer
    
    protected LossFunction loss;
    
    protected int totalNumberParams;
    
    protected float lambda;
    
    protected boolean dropoutActive = false;
    
    public abstract Tensor forwardPass(Tensor input) throws DimensionMismatchException;
    public abstract void backwardPass(Tensor input, Tensor expectation) throws DimensionMismatchException;
    
    public float calculateLoss(Tensor input, Tensor expectation) throws DimensionMismatchException {
        return loss.apply(forwardPass(input), expectation);
    }
    
    public void learn(Optimizer optimizer) throws DimensionMismatchException {
        
        optimizer.prepareLearningStep(gradient);
        
        for (Layer layer : layers) {
            layer.learn(gradient);
        }
        
        gradient = new Vector(gradient.getSize());
    
    }
    
    public void trainOnBatch(Tensor[] inputs, Tensor[] expectations, Optimizer optimizer) throws DimensionMismatchException {
        
        if (inputs.length != expectations.length || optimizer.getBatchSize() != inputs.length) {
            throw new IllegalArgumentException("Batch size mismatch");
        }
        
        for (int i = 0; i < optimizer.getBatchSize(); i++) {
            backwardPass(inputs[i], expectations[i]);
        }
        
        learn(optimizer);
        
    }
    
    public int getGradientSize() {
        return gradient.getSize();
    }
    
    public float getGradientAt(int parameterIndex) {
        return gradient.getValue(parameterIndex);
    }
    
    public int getNumberOfParameters() {
        return totalNumberParams;
    }
    
    public float getParameterAt(int parameterIndex) {
        
        int gradientIndex = layers[gradientLayerMapping[parameterIndex]].gradientIndex;
        
        return layers[gradientLayerMapping[parameterIndex]].getParameterAt(parameterIndex - gradientIndex);
    }
    
    protected void setParameterAt(int parameterIndex, float valueChange) {
        
        int gradientIndex = layers[gradientLayerMapping[parameterIndex]].gradientIndex;
        layers[gradientLayerMapping[parameterIndex]].setParameterAt(parameterIndex - gradientIndex, valueChange);
        
    }
    
    public float estimateGradient(int parameterIndex, float valueChange, Tensor input, Tensor expectation) throws DimensionMismatchException {
        
        float lossBefore = calculateLoss(input, expectation);
        
        float cached = getParameterAt(parameterIndex);
        setParameterAt(parameterIndex, cached + valueChange);
        
        float lossAfter = calculateLoss(input, expectation);
        
        setParameterAt(parameterIndex, cached);
        
        return (lossAfter - lossBefore) / valueChange;
        
    }
    
    public int[] getInputDimensions() {  
        return layers[0].getInputDimensions();
    }
    
    public int[] getOutputDimensions() {
        return layers[layers.length - 1].getOutputDimensions();
    }
    
    public abstract void saveToFile(File file) throws IOException;  
    
    public void toggleDropout(boolean dropoutActive) {
        this.dropoutActive = dropoutActive;
    }
    
}
