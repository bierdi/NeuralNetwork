package activationfunctions;

import tensors.Tensor;

/**
 *
 * @author bierdi
 */
public class SoftMax extends ActivationFunction {

    @Override
    public Tensor apply(Tensor t) {
        
        float max = Float.MIN_VALUE;
        
        for (int i = 0; i < t.getSize(); i++) {
            if (t.getValueFlat(i) > max) {
                max = t.getValueFlat(i);
            }
        }
        
        float sum = 0.0f;
        
        for (int i = 0; i < t.getSize(); i++) {
            sum += Math.exp(t.getValueFlat(i) - max);
        }
        
        for (int i = 0; i < t.getSize(); i++) {
            t.setValueFlat((float) Math.exp(t.getValueFlat(i) - max) / sum, i);
        }
        
        return t;
    }

    @Override
    public void derivative(Tensor t, Tensor gradient) {
        
        Tensor derivative = new Tensor(t.getDimensions());
        
        float max = Float.MIN_VALUE;
        
        for (int i = 0; i < t.getSize(); i++) {
            if (t.getValueFlat(i) > max) {
                max = t.getValueFlat(i);
            }
        }
        
        float sum = 0.0f;
        
        for (int i = 0; i < t.getSize(); i++) {
            sum += Math.exp(t.getValueFlat(i) - max);
        }
        
        for (int i = 0; i < t.getSize(); i++) {
            
            derivative.setValueFlat(((float) Math.exp(t.getValueFlat(i) - max) / sum + (float) Math.exp(t.getValueFlat(i) - max) * (float) Math.exp(t.getValueFlat(i) - max) * -1 / (sum * sum)) * gradient.getValueFlat(i), i);
            
            for (int j = 0; j < t.getSize(); j++) {
                
                if (j != i) {
                    derivative.setValueFlat(derivative.getValueFlat(i) + (-1 / (sum * sum) * (float) Math.exp(t.getValueFlat(i) - max) * (float) Math.exp(t.getValueFlat(j) - max)) * gradient.getValueFlat(j), i);
                }
                
            }
            
        }
        
        for (int i = 0; i < t.getSize(); i++) {
            gradient.setValueFlat(derivative.getValueFlat(i), i);
        }
        
    }
    
}
