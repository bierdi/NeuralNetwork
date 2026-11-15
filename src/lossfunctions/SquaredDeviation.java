package lossfunctions;

import exceptions.DimensionMismatchException;
import tensors.Tensor;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class SquaredDeviation extends LossFunction {

    @Override
    public float apply(Tensor result, Tensor expectation) throws DimensionMismatchException {
        
        checkInputs(result, expectation);
        
        float sum = 0;
        
        for (int i = 0; i < result.getSize(); i++) {
            sum += Math.pow(expectation.getValueFlat(i) - result.getValueFlat(i), 2) / 2;
        }
        
        return sum;
    }

    @Override
    public Tensor derivative(Tensor result, Tensor expectation) throws DimensionMismatchException {
        
        checkInputs(result, expectation);
        
        Tensor derivative = new Vector(result.getSize());
        
        for (int i = 0; i < result.getSize(); i++) {
            
            derivative.setValueFlat(result.getValueFlat(i) - expectation.getValueFlat(i), i);
            
        }
        
        return derivative;
    }
    
}
