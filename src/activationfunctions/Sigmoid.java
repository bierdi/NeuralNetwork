package activationfunctions;

/**
 *
 * @author bierdi
 */
public class Sigmoid implements ActivationFunction {

    @Override
    public double apply(double d) {
        return 1 / (1 + Math.exp(-d));
    }

    @Override
    public double applyDerivative(double d) {
        return Math.exp(-d) / ((1 + Math.exp(-d)) * (1 + Math.exp(-d)));
    }
    
}
