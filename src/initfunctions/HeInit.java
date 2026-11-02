package initfunctions;

import java.util.Random;

/**
 *
 * @author bierdi
 */
public class HeInit implements InitFunction {

    private final Random r;
    
    public HeInit(int seed) {
        r = new Random(seed);
    }
    
    @Override
    public float generateValue(int... params) {
        
        if (params == null || params.length < 1) {
            throw new IllegalArgumentException("Illegal number of parameters");
        }
        
        float a = (float) Math.sqrt(6.0f / params[0]);
        
        return (r.nextFloat() * 2 * a) - a;
    }
    
}
