package optimizers;

import tensors.Vector;

/**
 *
 * @author bierdi
 */
public abstract class Optimizer {
    
    protected float learningRate;
    protected int batchSize;
    
    public Optimizer(float learningRate, int batchSize) {
        this.learningRate = learningRate;
        this.batchSize = batchSize;
    }
    
    public abstract void prepareLearningStep(Vector gradient);
    
    public int getBatchSize() {
        return batchSize;
    }
    
}
