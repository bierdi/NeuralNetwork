package tensors;

import exceptions.DimensionMismatchException;

/**
 *
 * @author bierdi
 */
public class Matrix extends Tensor {
    
    private int width;
    private int height;

    public Matrix(float[][] values) {

        flattenValues(values);
        this.dimensions = new int[] {width, height};
        this.size = width * height;
        
        calculateOffsets(this.dimensions);

        this.startIndex = 0;

    }

    public Matrix(int width, int height) {

        super(new int[] {width, height});

        this.width = width;
        this.height = height;

    }

    private Matrix() {}

    private void flattenValues(float[][] values) {

        if (values == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (values.length == 0) {
            throw new IllegalArgumentException("Height must be greater 0");
        }

        if (values[0] == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (values[0].length == 0) {
            throw new IllegalArgumentException("Width must be greater 0");
        }

        this.height = values.length;
        this.width = values[0].length;
        this.values = new float[this.height * this.width];

        for (int i = 0; i < this.height; i++) {
            for (int j = 0; j < this.width; j++) {
                if (values[i] == null) {
                    throw new IllegalArgumentException("No null parameters allowed");
                }
                if (values[i].length != this.width) {
                    throw new IllegalArgumentException("Dimensions do not fit");
                }
                this.values[i * this.width + j] = values[i][j];
            }
        }

    }

    public static Matrix tensorToMatrix(Tensor t) {

        if (t == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (t.getDimensions().length != 2) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        Matrix m = new Matrix();
        m.values = t.values;
        m.size = t.size;
        m.dimensions = t.dimensions;
        m.offsets = t.offsets;
        m.width = t.dimensions[0];
        m.height = t.dimensions[1];
        m.startIndex = t.startIndex;
        m.isView = t.isView;

        return m;
    }

    public static Vector multiplyMatrixVector(Matrix m, Vector v) {

        if (m == null || v == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (m.getWidth() != v.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        Vector result = new Vector(m.getHeight());

        for (int i = 0; i < m.getHeight(); i++) {
            float sum = 0.0f;
            for (int j = 0; j < v.getSize(); j++) {
                sum += m.getValue(j, i) * v.getValue(j);
            }
            result.setValue(sum, i);
        }

        return result;
    }

    public static Matrix multiplyMatrices(Matrix m0, Matrix m1) {

        if (m0 == null || m1 == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (m1.getHeight() != m0.getWidth()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        Matrix result = new Matrix(m1.getWidth(), m0.getHeight());

        for (int i = 0; i < m1.getWidth(); i++) {
            for (int j = 0; j < m0.getHeight(); j++) {
                float sum = 0.0f;
                for (int k = 0; k < m1.getHeight(); k++) {
                    sum += m0.getValue(k, j) * m1.getValue(i, k);
                }
                result.setValue(sum, i, j);
            }
        }

        return result;
    }
    
    public void addMatrix(Matrix m) throws DimensionMismatchException {
        
        if (m == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (this.getHeight() != m.getHeight() || this.getWidth() != m.getWidth()) {
            throw new DimensionMismatchException("Dimensions do not fit");
        }
        
        for (int i = 0; i < m.getWidth(); i++) {
            for (int j = 0; j < m.getHeight(); j++) {
                this.setValueUnchecked(this.getValueUnchecked(i, j) + m.getValueUnchecked(i, j), i, j);
            }
        }
        
    }
    
    public static Matrix addMatrices(Matrix m0, Matrix m1) throws DimensionMismatchException {
        
        if (m0 == null || m1 == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }
        
        if (m0.getHeight() != m1.getHeight() || m0.getWidth() != m1.getWidth()) {
            throw new DimensionMismatchException("Dimensions do not fit");
        }
        
        Matrix result = new Matrix(m0.getWidth(), m0.getHeight());
        
        for (int i = 0; i < m0.getWidth(); i++) {
            for (int j = 0; j < m0.getHeight(); j++) {
                result.setValueUnchecked(m0.getValueUnchecked(i, j) + m1.getValueUnchecked(i, j), i, j);
            }
        }
        
        return result;
    }

    public static Matrix createFromVectors(Vector v0, Vector v1) {

        if (v0 == null || v1 == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        Matrix result = new Matrix(v0.getSize(), v1.getSize());

        for (int i = 0; i < result.getHeight(); i++) {
            for (int j = 0; j < result.getWidth(); j++) {
                result.setValue(v0.getValue(j) * v1.getValue(i), j, i);
            }
        }

        return result;
    }

    public static Matrix transposeMatrix(Matrix m) {

        if (m == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        Matrix result = new Matrix(m.getHeight(), m.getWidth());

        for (int i = 0; i < m.getWidth(); i++) {
            for (int j = 0; j < m.getHeight(); j++) {
                result.setValue(m.getValue(i, j), j, i);
            }
        }

        return result;
    }

    public float getValue(int x, int y) {

        if (x < 0 || x >= width || y < 0 || y >= height) {
            throw new IllegalArgumentException("Position out of bounds");
        }

        return this.values[startIndex + y * this.offsets[1] + x * this.offsets[0]];
    }
    
    private float getValueUnchecked(int x, int y) {
        return this.values[startIndex + y * this.offsets[1] + x * this.offsets[0]];
    }

    public void setValue(float value, int x, int y) {

        if (x < 0 || x >= width || y < 0 || y >= height) {
            throw new IllegalArgumentException("Position out of bounds");
        }

        this.values[startIndex + y * this.offsets[1] + x * this.offsets[0]] = value;
    }
    
    private void setValueUnchecked(float value, int x, int y) {
        this.values[startIndex + y * this.offsets[1] + x * this.offsets[0]] = value;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

}
