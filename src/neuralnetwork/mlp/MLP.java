package neuralnetwork.mlp;

import activationfunctions.ActivationFunction;
import exceptions.DimensionMismatchException;
import initfunctions.InitFunction;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.InvocationTargetException;
import java.util.Scanner;
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
    
    public MLP(int[] layerSizes, int inputSize, ActivationFunction[] activators, InitFunction[] inits, LossFunction loss, float lambda, float[] dropoutRates) {
        
        if (layerSizes == null || activators == null || loss == null || dropoutRates == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (layerSizes.length == 0) {
            throw new IllegalArgumentException("Network must have at least one layer");
        }
        
        if (layerSizes.length != activators.length || layerSizes.length != dropoutRates.length) {
            throw new IllegalArgumentException("Each layer must have exactly one activator and a dropout rate");
        }
        
        this.loss = loss;
        this.lambda = lambda;
        
        layers = new Layer[layerSizes.length * 2];
        
        totalNumberParams = 0;
        
        for (int i = 0; i < layers.length; i += 2) {
            layers[i] = new MLPLayer(layerSizes[i / 2], inputSize, activators[i / 2], inits[i / 2]);
            layers[i + 1] = new MLPDropoutLayer(dropoutRates[i / 2], layers[i].getOutputDimensions()[0]);
            totalNumberParams += layers[i].getNumberOfParameters();
            inputSize = layerSizes[i / 2];
        }
        
        int gradientIndex = 0;
        gradientLayerMapping = new int[totalNumberParams];
        gradient = new Vector(totalNumberParams);
        
        for (int i = 0; i < layers.length; i += 2) {
            
            int numParams = layers[i].getNumberOfParameters();
            layers[i].gradientIndex = gradientIndex;
            
            for (int j = gradientIndex; j < gradientIndex + numParams; j++) {
                gradientLayerMapping[j] = i;
            }
            
            gradientIndex += numParams;
            
        }
        
    }
    
    public MLP(Layer[] layers, LossFunction loss, float lambda) throws DimensionMismatchException {
        
        if (layers == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (layers.length == 0) {
            throw new IllegalArgumentException("Network must have at least one layer");
        }
        
        buildFromLayers(layers, loss, lambda);
        
    }
    
    private void buildFromLayers(Layer[] layers, LossFunction loss, float lambda) throws DimensionMismatchException {
        
        if (!(layers[0] instanceof MLPLayer) && !(layers[0] instanceof MLPDropoutLayer)) {
            throw new IllegalArgumentException("Type of layer not accepted in MLP");
        }
        
        totalNumberParams = layers[0].getNumberOfParameters();
        this.lambda = lambda;
        
        for (int i = 1; i < layers.length; i++) {
            
            if (!(layers[i] instanceof MLPLayer) && !(layers[i] instanceof MLPDropoutLayer)) {
                throw new IllegalArgumentException("Type of layer not accepted in MLP");
            }
        
            totalNumberParams += layers[i].getNumberOfParameters();
            
            int[] outputDimensions = layers[i - 1].getOutputDimensions();
            int[] inputDimensions = layers[i].getInputDimensions();
            
            if (outputDimensions.length != inputDimensions.length) {
                throw new DimensionMismatchException("Dimensions do not fit");
            }
            
            for (int j = 0; j < outputDimensions.length; j++) {
                if (outputDimensions[j] != inputDimensions[j]) {
                    throw new DimensionMismatchException("Dimensions do not fit");
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
    
    public MLP(File source) throws FileNotFoundException, ClassNotFoundException, NoSuchMethodException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, DimensionMismatchException {
        
        Scanner scanner = new Scanner(source);
        
        int numLayers = scanner.nextInt();
        Layer[] tmpLayers = new Layer[numLayers];
        
        String lossFunction = scanner.nextLine();
        
        Class<?> c = Class.forName(lossFunction);
        LossFunction tmpLoss = (LossFunction) c.getConstructor((Class<?>) null).newInstance();
        
        float lambda = scanner.nextFloat();
        
        for (int i = 0; i < numLayers; i++) {
            
            //MLPLayer l = new MLPLayer(scanner);
            
            /*
            String layerType = scanner.nextLine();
            String activationFunction = scanner.nextLine();
            
            int width = scanner.nextInt();
            int height = scanner.nextInt();
            
            Matrix tmpWeights = new Matrix(width, height);
            Vector tmpBiases = new Vector(height);
            
            c = Class.forName(layerType);
            Layer tmpLayer = (Layer) c.getConstructor((Class<?>) null).newInstance();
            */

        }
        
        buildFromLayers(tmpLayers, tmpLoss, lambda);
        
    }

    @Override
    public Vector forwardPass(Tensor input) throws DimensionMismatchException {
        
        if (input == null) {
            throw new IllegalArgumentException("Input must not be null");
        }
        
        if (input.getSize() != getInputDimensions()[0]) {
            throw new DimensionMismatchException("Input dimensions wrong");
        }
        
        Tensor inOut = input.deepCopy();
        
        for (Layer layer : layers) {
            inOut = layer.forwardPass(inOut, this.dropoutActive);
        }
        
        return Vector.tensorToVector(inOut);
    }
    
    @Override
    public void backwardPass(Tensor input, Tensor expectation) throws DimensionMismatchException {

        for (int i = 0; i < this.getNumberOfParameters(); i++) {
            this.gradient.setValue(this.gradient.getValue(i) + this.getParameterAt(i) * 2 * lambda, i);
        }
        
        Tensor lossDerivative = loss.derivative(forwardPass(input), expectation);
        
        for (int i = layers.length - 1; i >= 0; i--) {
            lossDerivative = layers[i].backwardPass(Vector.tensorToVector(lossDerivative), gradient);
        }
        
    }

    @Override
    public void saveToFile(File file) throws IOException {
        
        StringBuilder sb = new StringBuilder();
        
        sb.append(this.layers.length);
        sb.append("\n");
        sb.append(this.loss.getLossFunctionName());
        sb.append("\n");
        sb.append(this.lambda);
        
        for (Layer layer : this.layers) {
            sb.append("\n");
            layer.appendToStringBuilder(sb);
        }
        
        Writer writer = new FileWriter(file);
        writer.write(sb.toString());
        writer.flush();
        writer.close();
        
    }
    
}
