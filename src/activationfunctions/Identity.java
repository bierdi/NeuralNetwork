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
    public void derivative(Tensor t, Tensor gradient) {
        for (int i = 0; i < t.getSize(); i++) {
            t.setValueFlat(1, i);
        }
    }
    
}
