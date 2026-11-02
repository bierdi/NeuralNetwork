package tensors;

import java.util.Arrays;

/**
 *
 * @author bierdi
 */
public class Tensor {
    
    protected int[] dimensions;
    protected float[] values;
    
    protected int startIndex;
    protected int endIndex;
    
    private int[] offsets;
    
    public Tensor(int[] dimensions, float[] values) {
        
        if (values == null) {
            throw new IllegalArgumentException("Values must not be null");
        }
        
        int product = calculateDimensionProduct(dimensions);
        
        if (values.length != product) {
            throw new IllegalArgumentException("Number of values does not fit given dimensions");
        }
        
        this.dimensions = dimensions.clone();
        this.values = values.clone();
        
        startIndex = 0;
        endIndex = values.length;
        
        calculateOffsets();
        
    }
    
    public Tensor(int[] dimensions, Tensor t, int from, int to) {
        
        if (t == null || t.values == null) {
            throw new IllegalArgumentException("Values must not be null");
        }
        
        if (from < 0 || to > t.getSize() || from >= to) {
            throw new IllegalArgumentException("Illegal bound parameters");
        }
        
        int product = calculateDimensionProduct(dimensions);
        
        if (to - from != product) {
            throw new IllegalArgumentException("Number of values does not fit given dimensions");
        }
        
        this.dimensions = dimensions.clone();
        this.values = t.values;
        
        startIndex = from + t.startIndex;
        endIndex = to + t.startIndex;
        
        calculateOffsets();
        
    }
    
    public Tensor(int[] dimensions) {
        
        int product = calculateDimensionProduct(dimensions);
        
        this.dimensions = dimensions.clone();
        values = new float[product];
        
        startIndex = 0;
        endIndex = values.length;
        
        calculateOffsets();
        
    }
    
    protected Tensor() {}
    
    private int calculateDimensionProduct(int[] dimensions) {
        
        if (dimensions == null) {
            throw new IllegalArgumentException("Dimensions must not be null");
        }
        
        if (dimensions.length < 1) {
            throw new IllegalArgumentException("Length of dimensions must be greater 0");
        }
        
        int product = 1;
        
        for (int i = 0; i < dimensions.length; i++) {
            
            if (dimensions[i] <= 0) {
                throw new IllegalArgumentException("Dimensions value must be greater 0");
            }
            
            product *= dimensions[i];
            
        }
        
        return product;
    }
    
    protected final void calculateOffsets() {
        
        offsets = new int[dimensions.length];
        
        offsets[0] = 1;
        int product = 1;
        
        for (int i = 1; i < dimensions.length; i++) {
            
            product *= dimensions[i - 1];
            offsets[i] = product;
            
        }
        
    }
    
    public float getValue(int... position) {
        
        if (position == null) {
            throw new IllegalArgumentException("Position must not be null");
        }
        
        if (position.length != dimensions.length) {
            throw new IllegalArgumentException("Position does not match tensor dimensions");
        }
        
        int index = getIndex(position);
        
        return values[index];
    }
    
    public void setValue(float value, int... position) {
        
        if (position == null) {
            throw new IllegalArgumentException("Position must not be null");
        }
        
        if (position.length != dimensions.length) {
            throw new IllegalArgumentException("Position does not match tensor dimensions");
        }
        
        int index = getIndex(position);
        
        values[index] = value;
    }
    
    private int getIndex(int[] position) {
        
        int index = startIndex;
        
        for (int i = 0; i < position.length; i++) {
           
            if (position[i] < 0 || position[i] >= dimensions[i]) {
                throw new IllegalArgumentException("Position does not match tensor dimensions");
            }
            
            index += position[i] * offsets[i];
            
        }
        
        return index;
    }
    
    public float getValueFlat(int index) {
        return values[index + startIndex];
    }
    
    public void setValueFlat(float value, int index) {
        values[index + startIndex] = value;
    }
    
    public void addTensor(Tensor t) {
        
        if (t == null) {
            throw new IllegalArgumentException("Tensor must not be null");
        }
        
        if (dimensions.length != t.dimensions.length) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        for (int i = 0; i < dimensions.length; i++) {
            if (dimensions[i] != t.dimensions[i]) {
                throw new IllegalArgumentException("Dimensions do not fit");
            }
        }
        
        for (int i = 0; i < getSize(); i++) {
            setValueFlat(t.getValueFlat(i) + getValueFlat(i), i);
        }
        
    }
    
    public static final Tensor addTensors(Tensor t0, Tensor t1) {
        
        if (t0 == null || t1 == null) {
            throw new IllegalArgumentException("Tensors must not be null");
        }
        
        if (t0.dimensions.length != t1.dimensions.length) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        for (int i = 0; i < t0.dimensions.length; i++) {
            if (t0.dimensions[i] != t1.dimensions[i]) {
                throw new IllegalArgumentException("Dimensions do not fit");
            }
        }
        
        Tensor result = new Tensor(t0.dimensions);
        
        for (int i = 0; i < t0.getSize(); i++) {
            result.setValueFlat(t0.getValueFlat(i) + t1.getValueFlat(i), i);
        }
        
        return result;
    }
    
    public Tensor deepCopy() {
        return new Tensor(dimensions, Arrays.copyOfRange(values, startIndex, endIndex));
    }
    
    public int[] getDimensions() {
        return dimensions.clone();
    }
    
    public int getSize() {
        return endIndex - startIndex;
    }
    
    @Override
    public String toString() {
        return Arrays.toString(Arrays.copyOfRange(values, startIndex, endIndex));
    }
    
}
