package tensors;

import java.util.Arrays;

/**
 *
 * @author bierdi
 */
public class Matrix extends Tensor {
    
    private final int width;
    private final int height;
    
    public Matrix(int width, int height) {
        
        super(new int[] {width, height});
        
        this.width = width;
        this.height = height;
        
    }
    
    public Matrix(float[][] values) {
        
        flattenValues(values);
        
        this.width = values[0].length;
        this.height = values.length;
        this.dimensions = new int[] {this.width, this.height};
        startIndex = 0;
        endIndex = width * height;
        
        calculateOffsets();

    }
    
    private Matrix(int width, int height, float[] values) {
        
        if (width * height != values.length) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        this.width = width;
        this.height = height;
        this.dimensions = new int[] {this.width, this.height};
        
        this.values = values.clone();
        
        startIndex = 0;
        endIndex = width * height;
        
        calculateOffsets();
        
    }
    
    public Matrix(int width, int height, Tensor t, int from, int to) {
        
        if (t == null || t.values == null) {
            throw new IllegalArgumentException("Values must not be null");
        }
        
        if (from < 0 || to > t.getSize() || from >= to) {
            throw new IllegalArgumentException("Illegal bound parameters");
        }
        
        int product = width * height;
        
        if (to - from != product) {
            throw new IllegalArgumentException("Number of values does not fit given dimensions");
        }
        
        this.dimensions = new int[] {width, height};
        this.values = t.values;
        this.width = width;
        this.height = height;
        
        startIndex = from + t.startIndex;
        endIndex = to + t.startIndex;
        
        calculateOffsets();
        
    }
    
    private void flattenValues(float[][] values) {
        
        if (values == null) {
            throw new IllegalArgumentException("Values must not be null");
        }
        
        if (values.length == 0) {
            throw new IllegalArgumentException("Number of columns must be greater 0");
        }
        
        if (values[0] == null) {
            throw new IllegalArgumentException("Row must not be null");
        }
            
        if (values[0].length == 0) {
            throw new IllegalArgumentException("Number of elements in row must be greater 0");
        }
        
        int valuesHeight = values[0].length;
        
        this.values = new float[values.length * valuesHeight];
        
        for (int i = 0; i < values.length; i++) {
            
            if (values[i] == null) {
                throw new IllegalArgumentException("Row must not be null");
            }
            
            if (values[i].length != valuesHeight) {
                throw new IllegalArgumentException("Dimensions do not fit");
            }
            
            System.arraycopy(values[i], 0, this.values, i * valuesHeight, values[i].length);
            
        }
        
    }
    
    public static Matrix createFromVectors(Vector rows, Vector rowsScalingFactors) {
        
        Matrix m = new Matrix(rows.getSize(), rowsScalingFactors.getSize());
        
        for (int i = 0; i < rowsScalingFactors.getSize(); i++) {
            for (int j = 0; j < rows.getSize(); j++) {
                m.setValueFlat(rows.getValue(j) * rowsScalingFactors.getValue(i), i * rows.getSize() + j);
            }
        }
        
        return m;
    }
    
    public static Vector multiplyMatrixVector(Matrix m, Vector v) {
        
        if (m.getWidth() != v.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        Vector result = new Vector(m.getHeight());
        
        for (int i = 0; i < m.getHeight(); i++) {
            float sum = 0;
            for (int j = 0; j < m.getWidth(); j++) {
                sum += m.getValue(j, i) * v.getValue(j);
            }
            result.setValue(sum, i);
        }
        
        return result;
    }
    
    public static Matrix multiplyMatrizes(Matrix m0, Matrix m1) {
        
        if (m0.getWidth() != m1.getHeight()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        Matrix result = new Matrix(m1.getWidth(), m0.getHeight());
        
        for (int i = 0; i < m1.getWidth(); i++) {
            for (int j = 0; j < m0.getHeight(); j++) {
                float sum = 0;
                for (int k = 0; k < m1.getHeight(); k++) {
                    sum += m1.getValue(i, k) * m0.getValue(k, j);
                }
                result.setValue(sum, i, j);
            }
        }
        
        return result;
    }
    
    public void addMatrix(Matrix m) {
        super.addTensor(m);
    }
    
    public static Matrix addMatrizes(Matrix m0, Matrix m1) {
        
        if (m0 == null || m1 == null) {
            throw new IllegalArgumentException("Matrizes must not be null");
        }
        
        if (m0.getWidth() != m1.getWidth() || m0.getHeight() != m1.getHeight()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        float[] resultValues = new float[m0.getSize()];
        
        for (int i = 0; i < m0.getSize(); i++) {
            resultValues[i] = m0.getValueFlat(i) + m1.getValueFlat(i);
        }
        
        return new Matrix(m0.getWidth(), m0.getHeight(), resultValues);
    }
    
    public static Matrix transposeMatrix(Matrix m) {
        
        Matrix result = new Matrix(m.getHeight(), m.getWidth());
        
        for (int i = 0; i < m.getWidth(); i++) {
            for (int j = 0; j < m.getHeight(); j++) {
                result.setValue(m.getValue(i, j), j, i);
            }
        }
        
        return result;
    }
    
    public int getWidth() {
        return width;
    }
    
    public int getHeight() {
        return height;
    }
    
    @Override
    public int getSize() {
        return width * height;
    }
    
    @Override
    public Matrix deepCopy() {
        return new Matrix(width, height, Arrays.copyOfRange(values, startIndex, endIndex));
    }
    
}
