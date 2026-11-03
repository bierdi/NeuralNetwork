package matrixvector;

import java.util.Arrays;

/**
 *
 * @author bierdi
 */
public class Matrix {
    
    private final int columns;
    private final int rows;
    
    private double[][] matrix;
    
    //create Matrix with given dimensions
    public Matrix(int columns, int rows) {
        
        if(columns==0 || rows==0) {
            throw new IllegalArgumentException("Dimensions must be greater 0");
        }
        
        this.columns = columns;
        this.rows = rows;
        
        matrix = new double[columns][rows];
    }
    
    //create Matrix from double[][]
    public Matrix(double[][] values) {
        
        if(values==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        int r;
        int c = values.length;
        if(values.length!=0 && values[0]!=null) {
            r = values[0].length;
        } else {
            r = 0;
        }
        
        if(!checkDimensions(r, c, values)) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        columns = c;
        rows = r;
        
        matrix = Arrays.copyOf(values, values.length);
        
    }
    
    private boolean checkDimensions(int rows, int columns, double[][] matrix) {
        
        if(rows==0 || columns==0 || columns!=matrix.length) return false;
        
        for(int i = 0; i<matrix.length; i++) {
            if(matrix[i]==null || matrix[i].length!=rows) return false;
        }
        
        return true;
    }
    
    public void setValue(int x, int y, double value) {
        matrix[x][y] = value;
    }
    
    public double getValue(int x, int y) {
        return matrix[x][y];
    }
    
    public static Vector multiplyVector(Matrix m, Vector v) {
        
        if(m==null || v==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if(m.getColumns()!=v.getSize()) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }
        
        Vector result = new Vector(m.getRows());
        
        for(int i = 0; i<m.getRows(); i++) {
            double sum = 0;
            for(int j = 0; j<m.getColumns(); j++) {
                sum += m.getValue(j, i) * v.getValue(j);
            }
            result.setValue(i, sum);
        }
        
        return result;
    }
    
    public static Matrix multiplyMatrix(Matrix x, Matrix y) {
        
        if(x==null || y==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if(x.getColumns()!=y.getRows()) {
            throw new IllegalArgumentException("Dimesions do not fit");
        }
        
        double[][] tmp = new double[y.getColumns()][x.getRows()];
        
        for(int i = 0; i<x.getRows(); i++) {
            for(int j = 0; j<y.getColumns(); j++) {
                double sum = 0;
                for(int k = 0; k<y.getRows(); k++) {
                    sum += x.getValue(k, i) * y.getValue(j, k);
                }
                tmp[j][i] = sum; 
            }
        }
        
        return new Matrix(tmp);
    }
    
    public void multiplyScalar(double s) {
        
        for(int i = 0; i<this.getColumns(); i++) {
            for(int j = 0; j<this.getRows(); j++) {
                this.setValue(i, j, this.getValue(i, j) * s);
            }
        }
        
    }
   
    public static Matrix multiplyScalar(Matrix m, double s) {
        
        if(m==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        Matrix result = new Matrix(m.getColumns(), m.getRows());
        
        for(int i = 0; i<m.getColumns(); i++) {
            for(int j = 0; j<m.getRows(); j++) {
                result.setValue(i, j, m.getValue(i, j) * s);
            }
        }
        
        return result;
    }
    
    public void addMatrix(Matrix m) {
        
        if(m==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if(this.getColumns()!=m.getColumns() || this.getRows()!=m.getRows()) {
            throw new IllegalArgumentException("Dimesions do not fit");
        }
        
        for(int i = 0; i<this.getColumns(); i++) {
            for(int j = 0; j<this.getRows(); j++) {
                this.setValue(i, j, this.getValue(i, j) + m.getValue(i, j));
            }
        }
        
    }
    
    public static Matrix addMatrix(Matrix x, Matrix y) {
        
        if(x==null || y==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        if(x.getColumns()!=y.getColumns() || x.getRows()!=y.getRows()) {
            throw new IllegalArgumentException("Dimesions do not fit");
        }
        
        Matrix result = new Matrix(x.getColumns(), x.getRows());
        
        for(int i = 0; i<x.getColumns(); i++) {
            for(int j = 0; j<x.getRows(); j++) {
                result.setValue(i, j, x.getValue(i, j) + y.getValue(i, j));
            }
        }
        
        return result;
    }
    
    public static Matrix transposeMatrix(Matrix m) {
        
        if(m==null) {
            throw new IllegalArgumentException("Parameters must not be null");
        }
        
        Matrix result = new Matrix(m.getRows(), m.getColumns());
        
        for(int i = 0; i<m.getColumns(); i++) {
            for(int j = 0; j<m.getRows(); j++) {
                result.setValue(j, i, m.getValue(i, j));
            }
        }
        
        return result;
    }
    
    public int getRows() {
        return rows;
    }
    
    public int getColumns() {
        return columns;
    }
    
}
