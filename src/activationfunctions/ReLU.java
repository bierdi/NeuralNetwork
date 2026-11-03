package activationfunctions;

import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public class ReLU extends ActivationFunction {

    @Override
    public Tensor apply(Tensor t) {
        
        for (int i = 0; i < t.getSize(); i++) {
            t.setValueFlat(applyScalar(t.getValueFlat(i)), i);
        }
        
        return t;
    }

    @Override
    public Tensor derivative(Tensor t) {
        
        for (int i = 0; i < t.getSize(); i++) {
            t.setValueFlat(derivativeScalar(t.getValueFlat(i)), i);
        }
        
        return t;
    }
    
    @Override
    protected float applyScalar(float f) {
        return f > 0 ? f : 0;
    }
    
    @Override
    protected float derivativeScalar(float f) {
        return f > 0 ? 1 : 0;
    }
    
}
