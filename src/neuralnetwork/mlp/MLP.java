package neuralnetwork.mlp;

import activationfunctions.ActivationFunction;
import exceptions.InvalidDataException;
import initfunctions.InitFunction;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import lossfunctions.LossFunction;
import neuralnetwork.Layer;
import neuralnetwork.Network;
import tensors.Tensor;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class MLP extends Network {
    
    public MLP(int[] layerSizes, int inputSize, ActivationFunction[] activators, InitFunction[] inits, LossFunction loss) {
        
        if (layerSizes == null || activators == null || loss == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (layerSizes.length == 0) {
            throw new IllegalArgumentException("Network must have at least one layer");
        }
        
        if (layerSizes.length != activators.length) {
            throw new IllegalArgumentException("Each layer must have exactly one activator");
        }
        
        this.loss = loss;
        
        layers = new MLPLayer[layerSizes.length];
        
        totalNumberParams = 0;
        
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new MLPLayer(layerSizes[i], inputSize, activators[i], inits[i]);
            totalNumberParams += layers[i].getNumberOfParameters();
            inputSize = layerSizes[i];
        }
        
        int gradientIndex = 0;
        gradientLayerMapping = new int[totalNumberParams];
        gradient = new Vector(totalNumberParams);
        
        for (int i = 0; i < layers.length; i++) {
            
            int numParams = layers[i].getNumberOfParameters();
            layers[i].gradientIndex = gradientIndex;
            
            for (int j = gradientIndex; j < gradientIndex + numParams; j++) {
                gradientLayerMapping[j] = i;
            }
            
            gradientIndex += numParams;
            
        }
        
    }
    
    public MLP(MLPLayer[] layers, LossFunction loss) {
        
        if (layers == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (layers.length == 0) {
            throw new IllegalArgumentException("Network must have at least one layer");
        }
        
        totalNumberParams = layers[0].getNumberOfParameters();
        
        for (int i = 1; i < layers.length; i++) {
        
            totalNumberParams += layers[i].getNumberOfParameters();
            
            int[] outputDimensions = layers[i - 1].getOutputDimensions();
            int[] inputDimensions = layers[i].getInputDimensions();
            
            if (outputDimensions.length != inputDimensions.length) {
                throw new IllegalArgumentException("Dimensions do not fit");
            }
            
            for (int j = 0; j < outputDimensions.length; j++) {
                if (outputDimensions[j] != inputDimensions[j]) {
                    throw new IllegalArgumentException("Dimensions do not fit");
                }
            }
            
        }
        
        int gradientIndex = 0;
        gradientLayerMapping = new int[totalNumberParams];
        gradient = new Vector(totalNumberParams);
        this.layers = new MLPLayer[layers.length];
        
        for (int i = 0; i < layers.length; i++) {
            
            this.layers[i] = layers[i].deepCopy();
            int numParams = layers[i].getNumberOfParameters();
            this.layers[i].gradientIndex = gradientIndex;
            
            for (int j = gradientIndex; j < gradientIndex + numParams; j++) {
                gradientLayerMapping[j] = i;
            }
            
            gradientIndex += numParams;
            
        }
        
        this.loss = loss;
        
    }
    
    public MLP(File source) throws FileNotFoundException, InvalidDataException, ClassNotFoundException, NoSuchMethodException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
        throw new UnsupportedOperationException("Not supported yet");
    }

    @Override
    public Vector forwardPass(Tensor input) {
        
        if (input == null) {
            throw new IllegalArgumentException("Input must not be null");
        }
        
        if (input.getSize() != getInputDimensions()[0]) {
            throw new IllegalArgumentException("Input dimensions wrong");
        }
        
        Tensor inOut = input.deepCopy();
        
        for (Layer layer : layers) {
            inOut = layer.forwardPass(inOut);
        }
        
        return Vector.tensorToVector(inOut);
    }
    
    @Override
    public void backwardPass(Tensor input, Tensor expectation) {
        
        Tensor lossDerivative = loss.derivative(forwardPass(input), expectation);
        
        for (int i = layers.length - 1; i >= 0; i--) {
            lossDerivative = layers[i].backwardPass(Vector.tensorToVector(lossDerivative), gradient);
        }
        
    }

    @Override
    public void saveToFile(File file) throws IOException {
        throw new UnsupportedOperationException("Not supported yet");
    }
    
}
