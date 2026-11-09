package activationfunctions;

import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public abstract class ActivationFunction {
    
    public abstract Tensor apply(Tensor t);
    public abstract Tensor derivative(Tensor t);
    
}
