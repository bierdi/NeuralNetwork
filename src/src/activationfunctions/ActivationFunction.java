package activationfunctions;

import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public abstract class ActivationFunction {
    
    public abstract Tensor apply(Tensor t);
    public abstract Tensor derivative(Tensor t);
    protected abstract float applyScalar(float f);
    protected abstract float derivativeScalar(float f);
    
}
