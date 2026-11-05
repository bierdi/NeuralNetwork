package optimizers;

import exceptions.DimensionMismatchException;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class GradientDescentMomentum extends Optimizer {

    private Vector parameterMomentums;
    private final float momentum;
    
    private final float startingLearningRate;
    private final float decayRate;
    private int step;
    private final int decayStepCap;
    
    public GradientDescentMomentum(float learningRate, int batchSize, float momentum, float decayRate, int decayStepCap) {
        
        super(learningRate, batchSize);
        
        this.momentum = momentum;
        this.step = 0;
        this.decayRate = decayRate;
        this.startingLearningRate = learningRate;
        this.decayStepCap = decayStepCap;
        
    }
    
    @Override
    public void prepareLearningStep(Vector gradient) {
        
        if (step < decayStepCap) {
            learningRate = startingLearningRate * (1 / (1 + this.decayRate * step));
            step++;
        }
        
        if (parameterMomentums == null || parameterMomentums.getSize() != gradient.getSize()) {
            parameterMomentums = new Vector(gradient.getSize());
        }
         
        parameterMomentums.multiplyScalar(momentum);
        gradient.multiplyScalar(-learningRate / batchSize);
        
        try {
            gradient.addVector(parameterMomentums);
        } catch (DimensionMismatchException ex) {
            System.out.println(ex.getMessage());
        }
        
        parameterMomentums = Vector.tensorToVector(gradient.deepCopy());
        
    }
    
}
