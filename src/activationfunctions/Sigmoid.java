package activationfunctions;

import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public class Sigmoid extends ActivationFunction {

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

    private float applyScalar(float f) {
        return 1 / (1 + (float) Math.exp(-f));
    }

    private float derivativeScalar(float f) {
        return -(1 / ((1 + (float) Math.exp(-f)) * (1 + (float) Math.exp(-f)))) * (float) -Math.exp(-f);
    }
    
}
