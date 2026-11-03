package activationfunctions;

/**
 *
 * @author bierdi
 */
public class Identity implements ActivationFunction {

    @Override
    public double apply(double d) {
        return d;
    }

    @Override
    public double applyDerivative(double d) {
        return 1;
    }
    
}
