package lossfunctions;

import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public abstract class LossFunction {
    
    public abstract float apply(Tensor result, Tensor expectation);
    public abstract Tensor derivative(Tensor result, Tensor expectation);
    
    public String getLossFunctionName() {
        return this.getClass().getSimpleName();
    }
    
}
