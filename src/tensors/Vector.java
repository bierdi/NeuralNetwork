package tensors;

import exceptions.DimensionMismatchException;

/**
 *
 * @author bierdi
 */
public class Vector extends Tensor {
    
    public Vector(float[] values) {
        super(new int[] {values.length}, values);
    }

    public Vector(int size) {
        super(new int[] {size});
    }

    private Vector() {}

    public static Vector tensorToVector(Tensor t) {

        if (t == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (t.getDimensions().length != 1) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        Vector v = new Vector();
        v.values = t.values;
        v.size = t.size;
        v.dimensions = t.dimensions;
        v.offsets = t.offsets;
        v.startIndex = t.startIndex;
        v.isView = t.isView;

        return v;
    }
    
    public void addVector(Vector v) throws DimensionMismatchException {
        
        if (v == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (v.getSize() != this.getSize()) {
            throw new DimensionMismatchException("Dimensions do not fit");
        }
        
        for (int i = 0; i < this.getSize(); i++) {
            this.setValueUnchecked(this.getValueUnchecked(i) + v.getValueUnchecked(i), i);
        }
        
    }
    
    public static Vector addVectors(Vector v0, Vector v1) throws DimensionMismatchException {
        
        if (v0 == null || v1 == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (v0.getSize() != v1.getSize()) {
            throw new DimensionMismatchException("Dimensions do not fit");
        }
        
        Vector result = new Vector(v0.getSize());
        
        for (int i = 0; i < v0.getSize(); i++) {
            result.setValueUnchecked(v0.getValueUnchecked(i) + v1.getValueUnchecked(i), i);
        }
        
        return result;
    }
    
    @Override
    public void multiplyScalar(float f) {
        
        for (int i = 0; i < this.getSize(); i++) {
            this.setValueUnchecked(this.getValueUnchecked(i) * f, i);
        }
        
    }

    public float getValue(int x) {
        if (x < 0 || x >= size) throw new IllegalArgumentException("Position out of bounds");
        return this.values[startIndex + x * offsets[0]];
    }
    
    private float getValueUnchecked(int x) {
        return this.values[startIndex + x * offsets[0]];
    }

    public void setValue(float value, int x) {
        if (x < 0 || x >= size) throw new IllegalArgumentException("Position out of bounds");
        this.values[startIndex + x * offsets[0]] = value;
    }
    
    private void setValueUnchecked(float value, int x) {
        this.values[startIndex + x * offsets[0]] = value;
    }

}
