package lossfunctions;

import matrixvector.Matrix;
import matrixvector.Vector;

/**
 *
 * @author bierdi
 */
public interface LossFunction {
    
    public double apply(Vector x, Vector y);
    public Matrix applyDerivative(Vector x, Vector y);
    
}
