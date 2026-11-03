package tensors;

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

    public float getValue(int x) {
        if (x < 0 || x >= size) throw new IllegalArgumentException("Position out of bounds");
        return this.values[startIndex + x * offsets[0]];
    }

    public void setValue(float value, int x) {
        if (x < 0 || x >= size) throw new IllegalArgumentException("Position out of bounds");
        this.values[startIndex + x * offsets[0]] = value;
    }

}
