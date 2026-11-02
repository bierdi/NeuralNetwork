package tensors;

import java.util.Arrays;

/**
 *
 * @author bierdi
 */
public class Vector extends Tensor {
    
    private final int size;
    
    public Vector(int size) {
        
        super(new int[] {size});
        
        this.size = size;
        
    }
    
    public Vector(float[] values) {
        
        if (values == null) {
            throw new IllegalArgumentException("Values must not be null");
        }
        
        if (values.length == 0) {
            throw new IllegalArgumentException("Size must be greater 0");
        }
        
        this.size = values.length;
        this.dimensions = new int[] {this.size};
        this.values = values.clone();
        
        startIndex = 0;
        endIndex = size;
        
        calculateOffsets();
        
    }
    
    public Vector(int size, Tensor t, int from, int to) {
        
        if (t == null || t.values == null) {
            throw new IllegalArgumentException("Values must not be null");
        }
        
        if (from < 0 || to > t.getSize() || from >= to) {
            throw new IllegalArgumentException("Illegal bound parameters");
        }
        
        if (to - from != size) {
            throw new IllegalArgumentException("Number of values does not fit given size");
        }
        
        this.dimensions = new int[] {size};
        this.values = t.values;
        this.size = size;
        
        startIndex = from + t.startIndex;
        endIndex = to + t.startIndex;
        
        calculateOffsets();
        
    }
    
    public void addVector(Vector v) {
        super.addTensor(v);
    }
    
    public void multiplyVector(Vector v) {
        
        if (v == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        
        if (v.getSize() != size) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        for (int i = 0; i < size; i++) {
            setValue(v.getValueFlat(i) * getValueFlat(i), i);
        }
        
    }
    
    public void multiplyScalar(float f) {
        
        for (int i = 0; i < size; i++) {
            setValue(getValue(i) * f, i);
        }
        
    }
    
    public static Vector addVectors(Vector v0, Vector v1) {
        
        if (v0 == null || v1 == null) {
            throw new IllegalArgumentException("Vectors must not be null");
        }
        
        if (v0.getSize() != v1.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        float[] resultValues = new float[v0.getSize()];
        
        for (int i = 0; i < v0.getSize(); i++) {
            resultValues[i] = v0.getValueFlat(i) + v1.getValueFlat(i);
        }
        
        return new Vector(resultValues);
    }
    
    @Override
    public int getSize() {
        return size;
    }
    
    @Override
    public Vector deepCopy() {
        return new Vector(Arrays.copyOfRange(values, startIndex, endIndex));
    }
    
}
