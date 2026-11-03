package lossfunctions;

import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public interface LossFunction {
    
    public float apply(Tensor result, Tensor expectation);
    public Tensor derivative(Tensor result, Tensor expectation);
    
}
