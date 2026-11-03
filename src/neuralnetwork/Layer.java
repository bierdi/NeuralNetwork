package neuralnetwork;

import activationfunctions.ActivationFunction;
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
    
    public abstract Tensor forwardPass(Tensor input);
    public abstract Tensor backwardPass(Tensor gradient, Vector globalGradient);
    public abstract void learn(Vector globalGradient);
    
    public abstract int[] getInputDimensions();
    public abstract int[] getOutputDimensions();
    
    public abstract int getNumberOfParameters();
    public abstract float getParameterAt(int parameterIndex);
    public abstract void setParameterAt(int parameterIndex, float valueChange);
    
    public abstract Layer deepCopy();
    
    public String getActivationFunctionName() {
        return activator.getClass().getCanonicalName();
    }
    
}
