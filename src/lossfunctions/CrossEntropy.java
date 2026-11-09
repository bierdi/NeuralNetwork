package lossfunctions;

import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public class CrossEntropy extends LossFunction {

    @Override
    public float apply(Tensor result, Tensor expectation) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public Tensor derivative(Tensor result, Tensor expectation) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}
