package matrixvector;

import java.util.Arrays;

/**
 *
 * @author bierdi
 */
public class Vector {
    
    private final double[] vector;
    
    public Vector(int m) {   
        if(m==0) {
            throw new IllegalArgumentException("Dimension must be greater 0");
        }
        vector = new double[m];
    }
    
    public Vector(double[] values) {
        
        if(values==null || values.length==0) {
            throw new IllegalArgumentException("Values must not be null or empty");
        }
        
        vector = Arrays.copyOf(values, values.length);
        
    }
    
    public void setValue(int x, double value) {
        vector[x] = value;
    }
    
    public double getValue(int x) {
        return vector[x];
    }
    
    public int getSize() {
        return vector.length;
    }
    
    public double scalarProduct(Vector v) {
        
        if(v==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if(v.getSize()!=this.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        double sum = 0;
        
        for(int i = 0; i<this.getSize(); i++) {
            sum += this.getValue(i) * v.getValue(i);
        }
        
        return sum;
    }
    
    public static double scalarProduct(Vector x, Vector y) {
        
        if(x==null || y==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if(x.getSize()!=y.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        double sum = 0;
        
        for(int i = 0; i<x.getSize(); i++) {
            sum += x.getValue(i) * y.getValue(i);
        }
        
        return sum;
    }
    
    public void multiplyScalar(double s) {
        
        for(int i = 0; i<getSize(); i++) {
            setValue(i, getValue(i) * s);
        }
        
    }
    
    public static Vector multiplyScalar(Vector v, double s) {
        
        if(v==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        Vector result = new Vector(v.getSize());
        
        for(int i = 0; i<v.getSize(); i++) {
            result.setValue(i, v.getValue(i) * s);
        }
        
        return result;
    }
    
    public void addVector(Vector v) {
        
        if(v==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if(v.getSize()!=this.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        for(int i = 0; i<this.getSize(); i++) {
            this.setValue(i, this.getValue(i) + v.getValue(i));
        }
        
    }
    
    public static Vector addVector(Vector x, Vector y) {
        
        if(x==null || y==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if(x.getSize()!=y.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        Vector result = new Vector(x.getSize());
        
        for(int i = 0; i<x.getSize(); i++) {
            result.setValue(i, x.getValue(i) + y.getValue(i));
        }
        
        return result;
    }
    
    public static Vector slice(Vector v, int from, int to) {
        return new Vector(Arrays.copyOfRange(v.vector, from, to));
    }
    
    public static Matrix transposeVector(Vector v) {
        
        if(v==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        Matrix result = new Matrix(v.getSize(), 1);
        
        for(int i = 0; i<v.getSize(); i++) {
            result.setValue(i, 0, v.getValue(i));
        }
        
        return result;
    }
    
    public static Vector squareComponents(Vector v) {
        
        if(v==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        Vector result = new Vector(v.getSize());
        
        for(int i = 0; i<v.getSize(); i++) {
            result.setValue(i, v.getValue(i)*v.getValue(i));
        }
        
        return result;
    }
    
    public void squareRootComponents() {
        
        for(int i = 0; i<getSize(); i++) {
            this.setValue(i, Math.sqrt(this.getValue(i)));
        }
        
    }
    
    public void addScalar(double s) {
        
        for(int i = 0; i<getSize(); i++) {
            this.setValue(i, this.getValue(i)+s);
        }
        
    }
    
    public void invertComponents() {
        
        for(int i = 0; i<getSize(); i++) {
            this.setValue(i, 1/this.getValue(i));
        }
        
    }
    
    public static Vector multiplyComponents(Vector x, Vector y) {
        
        if(x==null || y==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if(x.getSize()!=y.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        Vector result = new Vector(x.getSize());
        
        for(int i = 0; i<x.getSize(); i++) {
            result.setValue(i, x.getValue(i)*y.getValue(i));
        }
        
        return result;
    }
    
    @Override
    public String toString() {
        return Arrays.toString(vector);
    }
    
}
