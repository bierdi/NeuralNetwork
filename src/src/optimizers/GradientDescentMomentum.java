package optimizers;

import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class GradientDescentMomentum extends Optimizer {

    public GradientDescentMomentum(float learningRate, int batchSize) {
        super(learningRate, batchSize);
    }
    
    @Override
    public void prepareLearningStep(Vector gradient) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}
