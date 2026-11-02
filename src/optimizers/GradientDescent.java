package optimizers;

import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class GradientDescent extends Optimizer {

    public GradientDescent(float learningRate, int batchSize) {
        super(learningRate, batchSize);
    }
    
    @Override
    public void prepareLearningStep(Vector gradient) {
        
        gradient.multiplyScalar(-learningRate / batchSize);
        
    }
    
}
