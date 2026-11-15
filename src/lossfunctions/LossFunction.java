package lossfunctions;

import exceptions.DimensionMismatchException;
import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public abstract class LossFunction {
    
    public abstract float apply(Tensor result, Tensor expectation) throws DimensionMismatchException;
    public abstract Tensor derivative(Tensor result, Tensor expectation) throws DimensionMismatchException;
    
    public String getLossFunctionName() {
        return this.getClass().getCanonicalName();
    }
    
    protected void checkInputs(Tensor result, Tensor expectation) throws DimensionMismatchException {
        
        if (result == null || expectation == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (result.getSize() != expectation.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        int[] resultDimensions = result.getDimensions();
        int[] expectationDimensions = expectation.getDimensions();
        
        if (resultDimensions.length != expectationDimensions.length) {
            throw new DimensionMismatchException("Dimensions do not fit");
        }
        
        for (int i = 0; i < resultDimensions.length; i++) {
            if (resultDimensions[i] != expectationDimensions[i]) {
                throw new DimensionMismatchException("Dimensions do not fit");
            }
        }
        
    }
    
}
