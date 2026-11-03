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
    public float generateValue(int... params) {
        return r.nextFloat() * 2 - 1;
    }
    
}
