package activationfunctions;

/**
 *
 * @author bierdi
 */
public class Relu implements ActivationFunction {

    @Override
    public double apply(double d) {
        return Math.max(d, 0);
    }

    @Override
    public double applyDerivative(double d) {
        return d > 0 ? 1 : 0;
    }
    
}
