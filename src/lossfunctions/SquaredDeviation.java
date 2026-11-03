package lossfunctions;

import matrixvector.Matrix;
import matrixvector.Vector;

/**
 *
 * @author bierdi
 */
public class SquaredDeviation implements LossFunction {

    @Override
    public double apply(Vector x, Vector y) {
        
        if (x == null || y == null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if (x.getSize() != y.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        double sum = 0;
        
        for (int i = 0; i < x.getSize(); i++) {
            sum += Math.pow(y.getValue(i) - x.getValue(i), 2) / 2.0;
        }
        
        return sum;
    }

    @Override
    public Matrix applyDerivative(Vector x, Vector y) {
        
        if (x==null || y==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if (x.getSize() != y.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        Matrix result = new Matrix(x.getSize(), 1);
        
        for (int i = 0; i < x.getSize(); i++) {
            result.setValue(i, 0, x.getValue(i) - y.getValue(i));
        }
        
        return result;
    }
        
}
