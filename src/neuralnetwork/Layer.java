package neuralnetwork;

import activationfunctions.ActivationFunction;
import tensors.Tensor;
import tensors.Vector;

/**
 *
 * @author bierdi
 * @param <W> Form of Weights
 * @param <B> Form of Biases
 * @param <I> Form of Input
 * @param <O> Form of Output
 */
public abstract class Layer<W extends Tensor, B extends Tensor, I extends Tensor, O extends Tensor> {
    
    protected W weights;
    protected B biases;
    
    public int gradientIndex; //lower bound index in global gradient
    
    protected I inputCache; //caches input for backward pass
    protected O tmpResultCache; //caches output before activation function
    
    protected ActivationFunction activator;
    
    public abstract O forwardPass(I input);
    public abstract I backwardPass(O gradient, Vector globalGradient);
    public abstract void learn(Vector gradient);
    
    public abstract int[] getInputDimensions();
    public abstract int[] getOutputDimensions();
    
    public abstract int getNumberOfParameters();
    public abstract float getParameterAt(int parameterIndex);
    public abstract void setParameterAt(int parameterIndex, float valueChange);
    
    public abstract Layer<W, B, I, O> deepCopy();
    
    public String getActivationFunctionName() {
        return activator.getClass().getCanonicalName();
    }
    
}
