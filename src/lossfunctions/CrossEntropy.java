package lossfunctions;

import exceptions.DimensionMismatchException;
import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public class CrossEntropy extends LossFunction {

    @Override
    public float apply(Tensor result, Tensor expectation) throws DimensionMismatchException {
        
        checkInputs(result, expectation);
        
        float sum = 0.0f;
        
        for (int i = 0; i < result.getSize(); i++) {
            
            if (expectation.getValueFlat(i) > 1 || expectation.getValueFlat(i) < 0 || result.getValueFlat(i) > 1 || result.getValueFlat(i) < 0) {
                throw new IllegalArgumentException("Result and expectation values must be probabilities between 0 and 1");
            }
            
            sum += expectation.getValueFlat(i) * Math.log(Math.min(Math.max(result.getValueFlat(i), 1E-7), 1));
            
        }
        
        return -sum;
    }

    @Override
    public Tensor derivative(Tensor result, Tensor expectation) throws DimensionMismatchException {
        
        checkInputs(result, expectation);
        
        Tensor derivative = new Tensor(result.getDimensions());
        
        for (int i = 0; i < result.getSize(); i++) {
            
            if (expectation.getValueFlat(i) > 1 || expectation.getValueFlat(i) < 0 || result.getValueFlat(i) > 1 || result.getValueFlat(i) < 0) {
                throw new IllegalArgumentException("Result and expectation values must be probabilities between 0 and 1");
            }
            
            derivative.setValueFlat(-expectation.getValueFlat(i) / (float) Math.min(Math.max(result.getValueFlat(i), 1E-7), 1), i);
            
        }
        
        return derivative;
    }
    
}
