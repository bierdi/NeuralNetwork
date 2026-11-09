package neuralnetwork.mlp;

import exceptions.DimensionMismatchException;
import java.util.Random;
import java.util.Scanner;
import neuralnetwork.Layer;
import tensors.Tensor;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class MLPDropoutLayer extends Layer {

    private final int inputSize;
    private final float dropoutRate;
    
    private final Random r;
    Vector dropoutCache;
    
    public MLPDropoutLayer(float dropoutRate, int inputSize) {
        
        if (inputSize < 1) {
            throw new IllegalArgumentException("Input size must be at least of size 1");
        }
        
        this.r = new Random();
        this.dropoutRate = dropoutRate;
        this.inputSize = inputSize;
        
    }
    
    @Override
    public Tensor forwardPass(Tensor input, boolean dropoutActive) throws DimensionMismatchException {
        
        Vector result = Vector.tensorToVector(input);
        this.dropoutCache = new Vector(this.inputSize);
        
        if (dropoutActive) {
            for (int i = 0; i < result.getSize(); i++) {
                float rand = r.nextFloat();
                if (rand < dropoutRate) {
                    result.setValue(0, i);
                } else {
                    this.dropoutCache.setValue(1 / (1 - this.dropoutRate), i);
                    result.setValue(result.getValue(i) * 1 / (1 - this.dropoutRate), i);
                }
            }
        } else {
            for (int i = 0; i < dropoutCache.getSize(); i++) {
                dropoutCache.setValue(1, i);  
            }
        }
        
        return result;
    }

    @Override
    public Tensor backwardPass(Tensor gradient, Vector globalGradient) throws DimensionMismatchException {
        Vector.tensorToVector(gradient).multiplyVectorValues(dropoutCache);
        return gradient;
    }

    @Override
    public void learn(Vector globalGradient) {}

    @Override
    public int[] getInputDimensions() {
        return new int[] {inputSize};
    }

    @Override
    public int[] getOutputDimensions() {
        return new int[] {inputSize};
    }

    @Override
    public int getNumberOfParameters() {
        return 0;
    }

    @Override
    public float getParameterAt(int parameterIndex) {
        throw new UnsupportedOperationException("This layer type has no parameters");
    }

    @Override
    public void setParameterAt(int parameterIndex, float valueChange) {
        throw new UnsupportedOperationException("This layer type has no parameters");
    }

    @Override
    public Layer deepCopy() throws DimensionMismatchException {
        return new MLPDropoutLayer(this.dropoutRate, this.inputSize);
    }

    @Override
    public void appendToStringBuilder(StringBuilder sb) {
        
        sb.append(this.getLayerName());
        sb.append("\n");
        sb.append(this.dropoutRate);
        sb.append("\n");
        sb.append(this.inputSize);   
        
    }

    @Override
    protected void buildFromFile(Scanner scanner) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
}
