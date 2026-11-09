package neuralnetwork;

import activationfunctions.ActivationFunction;
import exceptions.DimensionMismatchException;
import java.util.Scanner;
import tensors.Tensor;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public abstract class Layer {
    
    protected Tensor weights;
    protected Tensor biases;
    
    public int gradientIndex; //lower bound index in global gradient
    
    protected Tensor inputCache; //caches input for backward pass
    protected Tensor tmpResultCache; //caches output before activation function
    
    protected ActivationFunction activator;
    
    public Layer(Scanner scanner) {
        buildFromFile(scanner);
    }
    
    public Layer() {}
    
    public abstract Tensor forwardPass(Tensor input, boolean dropoutActive) throws DimensionMismatchException;
    public abstract Tensor backwardPass(Tensor gradient, Vector globalGradient) throws DimensionMismatchException;
    public abstract void learn(Vector globalGradient) throws DimensionMismatchException;
    
    public abstract int[] getInputDimensions();
    public abstract int[] getOutputDimensions();
    
    public abstract int getNumberOfParameters();
    public abstract float getParameterAt(int parameterIndex);
    public abstract void setParameterAt(int parameterIndex, float valueChange);
    
    public abstract Layer deepCopy() throws DimensionMismatchException;
    
    public String getActivationFunctionName() {
        if (activator == null) return null;
        return activator.getClass().getSimpleName();
    }
    
    public String getLayerName() {
        return this.getClass().getSimpleName();
    }
    
    public abstract void appendToStringBuilder(StringBuilder sb);
    protected abstract void buildFromFile(Scanner scanner);
    
}
