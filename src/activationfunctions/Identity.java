package activationfunctions;

import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public class Identity extends ActivationFunction {

    @Override
    public Tensor apply(Tensor t) {
        return t;
    }

    @Override
    public Tensor derivative(Tensor t) {
        for (int i = 0; i < t.getSize(); i++) {
            t.setValueFlat(1, i);
        }
        return t;
    }

    private float applyScalar(float f) {
        return f;
    }

    private float derivativeScalar(float f) {
        return 1;
    }
    
}
