package initfunctions;

import java.util.Random;

/**
 *
 * @author bierdi
 */
public class RandomInit implements InitFunction {

    private final Random r;
    
    public RandomInit(int seed) {
        r = new Random(seed);
    }
    
    @Override
    public double generate() {
        return r.nextDouble() * 2 - 1;
    }
    
}
