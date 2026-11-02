package neuralnetwork;

import java.io.File;
import java.io.IOException;
import lossfunctions.*;
import optimizers.Optimizer;
import tensors.Tensor;
import tensors.Vector;

/**
 *
 * @author bierdi
 * @param <I> Form of Input
 * @param <O> Form of Output
 */
public abstract class Network<I extends Tensor, O extends Tensor> {
    
    protected Layer[] layers;
    
    protected Vector gradient; //holds partial derivatives of all parameters
    protected int[] gradientLayerMapping; //maps entry in gradient to corresponding layer
    
    protected LossFunction loss;
    
    protected int totalNumberParams;
    
    public abstract O forwardPass(I input);
    public abstract void backwardPass(I input, O expectation);
    
    public float calculateLoss(I input, O expectation) {
        return loss.apply(forwardPass(input), expectation);
    }
    
    public void learn(Optimizer optimizer) {
        
        optimizer.prepareLearningStep(gradient);
        
        for (Layer layer : layers) {
            layer.learn(gradient);
        }
        
        gradient.multiplyScalar(0);
    
    }
    
    public void trainOnBatch(I[] inputs, O[] expectations, Optimizer optimizer) {
        
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
    
    public float estimateGradient(int parameterIndex, float valueChange, I input, O expectation) {
        
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
    
}
