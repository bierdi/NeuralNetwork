package tensors;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author bierdi
 */
public class Tensor {

    protected int[] dimensions;
    protected float[] values;

    protected int size;

    protected int[] offsets;
    protected int startIndex;
    
    protected boolean isView = false;

    public Tensor(int[] dimensions, float[] values) {

        if (dimensions == null || values == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        int product = calculateDimensionProduct(dimensions);

        if (product != values.length) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        calculateOffsets(dimensions);

        this.dimensions = dimensions.clone();
        this.values = values.clone();
        this.size = product;
        this.startIndex = 0;

    }

    public Tensor(int[] dimensions) {

        if (dimensions == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        int product = calculateDimensionProduct(dimensions);
        calculateOffsets(dimensions);

        this.dimensions = dimensions.clone();
        this.size = product;
        this.values = new float[this.size];
        this.startIndex = 0;

    }

    protected Tensor() {};

    private int calculateDimensionProduct(int[] dimensions) {

        if (dimensions.length == 0) {
            throw new IllegalArgumentException("Dimensions length must be greater 0");
        }

        int product = 1;

        for (int i = 0; i < dimensions.length; i++) {
            if (dimensions[i] <= 0) {
                throw new IllegalArgumentException("Each dimension must be greater 0");
            }
            product *= dimensions[i];
        }

        return product;
    }

    protected final void calculateOffsets(int[] dimensions) {

        if (dimensions.length == 0) {
            throw new IllegalArgumentException("Dimensions length must be greater 0");
        }

        offsets = new int[dimensions.length];
        offsets[0] = 1;

        for (int i = 1; i < dimensions.length; i++) {
            offsets[i] = offsets[i - 1] * dimensions[i - 1];
        }

    }

    public static Tensor createView(int[] dimensions, Tensor t, int[] axis, int[] anchorPoint) {

        if (dimensions == null || t == null || axis == null || anchorPoint == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (dimensions.length != axis.length || anchorPoint.length != t.dimensions.length) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        Map<Integer, Integer> map = new HashMap<>();
        Map<Integer, Integer> helper = new HashMap<>();

        Tensor result = new Tensor();
        result.dimensions = dimensions.clone();
        result.values = t.values;
        result.size = result.calculateDimensionProduct(dimensions);
        result.offsets = new int[dimensions.length];
        result.startIndex = t.calculateIndex(anchorPoint);
        result.isView = true;

        if (result.size > t.size) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        for (int i = 0; i < axis.length; i++) {

            if (axis[i] == 0) {
                throw new IllegalArgumentException("Axis indices must not be 0");
            }
            
            if (map.containsKey(-axis[i])) {
                throw new IllegalArgumentException("Negative counterpart of axis not allowed");
            }
            
            if (!map.containsKey(axis[i])) {

                result.offsets[i] = t.offsets[Math.abs(axis[i]) - 1] * (int) Math.signum((float) (axis[i]));
                map.put(axis[i], result.offsets[i] * dimensions[i]);
                helper.put(axis[i], dimensions[i] * (int) Math.signum((float) (axis[i])));

            } else if (map.containsKey(axis[i])) {

                result.offsets[i] = map.get(axis[i]);
                map.put(axis[i], result.offsets[i] * dimensions[i]);
                helper.put(axis[i], helper.get(axis[i]) * dimensions[i]);

            }

        }

        for (int i = 0; i < t.dimensions.length; i++) {

            if (map.containsKey(i + 1)) {
                int tmp = anchorPoint[i] + helper.get(i + 1);
                if (tmp < -1 || tmp > t.dimensions[i]) {
                    throw new IllegalArgumentException("Dimensions do not fit");
                }   
            } else if (map.containsKey(-(i + 1))) {
                int tmp = anchorPoint[i] + helper.get(-(i + 1));
                if (tmp < -1 || tmp > t.dimensions[i]) {
                    throw new IllegalArgumentException("Dimensions do not fit");
                }  
            }

        }

        return result;
    }

    public void addTensor(Tensor t) {

        if (t == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (!dimensionCheck(this.dimensions, t.dimensions)) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        addTensorRec(new int[this.dimensions.length], t, 0);

    }

    private void addTensorRec(int[] position, Tensor t, int i) {

        if (i == dimensions.length) {
            this.setValue(this.getValue(position) + t.getValue(position), position);
            return;
        }

        for (int j = 0; j < dimensions[i]; j++) {
            position[i] = j;
            addTensorRec(position, t, i + 1);
        }

    }

    public static Tensor addTensors(Tensor t0, Tensor t1) {

        if (t0 == null || t1 == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (!dimensionCheck(t0.dimensions, t1.dimensions)) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        Tensor result = new Tensor(t0.dimensions);

        addTensorsRec(result, t0, t1, new int[result.dimensions.length], 0);

        return result;
    }

    private static void addTensorsRec(Tensor result, Tensor t0, Tensor t1, int[] position, int i) {

        if (i == result.dimensions.length) {
            result.setValue(t0.getValue(position) + t1.getValue(position), position);
            return;
        }

        for (int j = 0; j < result.dimensions[i]; j++) {
            position[i] = j;
            addTensorsRec(result, t0, t1, position, i + 1);
        }

    }

    public void multiplyTensorValues(Tensor t) {

        if (t == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (!dimensionCheck(this.dimensions, t.dimensions)) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        multiplyTensorValuesRec(new int[dimensions.length], t, 0);

    }

    private void multiplyTensorValuesRec(int[] position, Tensor t, int i) {

        if (i == dimensions.length) {
            this.setValue(this.getValue(position) * t.getValue(position), position);
            return;
        }

        for (int j = 0; j < dimensions[i]; j++) {
            position[i] = j;
            multiplyTensorValuesRec(position, t, i + 1);
        }

    }

    public static Tensor multiplyTensorValues(Tensor t0, Tensor t1) {

        if (t0 == null || t1 == null) {
            throw new IllegalArgumentException("No null parameters allowed");
        }

        if (!dimensionCheck(t0.dimensions, t1.dimensions)) {
            throw new IllegalArgumentException("Dimensions do not fit");
        }

        Tensor result = new Tensor(t0.dimensions);

        multiplyTensorValuesRec(result, t0, t1, new int[result.dimensions.length], 0);

        return result;
    }

    private static void multiplyTensorValuesRec(Tensor result, Tensor t0, Tensor t1, int[] position, int i) {

        if (i == result.dimensions.length) {
            result.setValue(t0.getValue(position) * t1.getValue(position), position);
            return;
        }

        for (int j = 0; j < result.dimensions[i]; j++) {
            position[i] = j;
            multiplyTensorValuesRec(result, t0, t1, position, i + 1);
        }

    }

    public void multiplyScalar(float f) {

        multiplyScalarRec(new int[dimensions.length], f, 0);

    }

    private void multiplyScalarRec(int[] position, float f, int i) {

        if (i == dimensions.length) {
            this.setValue(this.getValue(position) * f, position);
            return;
        }

        for (int j = 0; j < dimensions[i]; j++) {
            position[i] = j;
            multiplyScalarRec(position, f, i + 1);
        }

    }

    public void addScalar(float f) {

        addScalarRec(new int[dimensions.length], f, 0);

    }

    private void addScalarRec(int[] position, float f, int i) {

        if (i == dimensions.length) {
            this.setValue(this.getValue(position) + f, position);
            return;
        }

        for (int j = 0; j < dimensions[i]; j++) {
            position[i] = j;
            addScalarRec(position, f, i + 1);
        }

    }

    protected static boolean dimensionCheck(int[] dimensions0, int[] dimensions1) {

        if (dimensions0 == null || dimensions1 == null) {
            return false;
        }

        if (dimensions0.length != dimensions1.length) {
            return false;
        }

        for (int i = 0; i < dimensions0.length; i++) {
            if (dimensions0[i] <= 0 || dimensions1[i] <= 0) {
                throw new IllegalArgumentException("Illegal dimensions");
            }
            if (dimensions0[i] != dimensions1[i]) {
                return false;
            }
        }

        return true;
    }

    public float getValue(int... position) {
        return values[calculateIndex(position)];
    }

    public void setValue(float value, int... position) {
        values[calculateIndex(position)] = value;
    }
    
    public float getValueFlat(int index) {
        if (isView) throw new UnsupportedOperationException("Not possible on a view");
        if (index < 0 || index > values.length) throw new IllegalArgumentException("Index out of bounds");
        return values[index];
    }
    
    public void setValueFlat(float value, int index) {
        if (isView) throw new UnsupportedOperationException("Not possible on a view");
        if (index < 0 || index > values.length) throw new IllegalArgumentException("Index out of bounds");
        values[index] = value;
    }

    private int calculateIndex(int... position) {

        if (position == null) {
            throw new IllegalArgumentException("Position must not be null");
        }

        if (position.length == 0) {
            throw new IllegalArgumentException("Position must have at least one dimension");
        }

        int index = startIndex;

        for (int i = 0; i < position.length; i++) {
            if (position[i] < 0 || position[i] >= dimensions[i]) throw new IllegalArgumentException("Position out of bounds");
            index += position[i] * offsets[i];
        }

        return index;
    }

    public int getSize() {
        return size;
    }

    public int[] getDimensions() {
        return dimensions.clone();
    }

    @Override
    public String toString() {

        float[] valueCopy = new float[this.size];

        valueCopyRec(new int[this.dimensions.length], valueCopy, this.dimensions.length - 1, new int[] {0});

        return Arrays.toString(valueCopy);
    }

    private void valueCopyRec(int[] position, float[] valueCopy, int i, int[] valueCopyIndex) {

        if (i == -1) {
            valueCopy[valueCopyIndex[0]] = this.getValue(position);
            valueCopyIndex[0]++;
            return;
        }

        for (int j = 0; j < dimensions[i]; j++) {
            position[i] = j;
            valueCopyRec(position, valueCopy, i - 1, valueCopyIndex);
        }

    }

    public Tensor deepCopy() {

        Tensor copy = new Tensor();

        copy.dimensions = this.dimensions.clone();
        
        if (this.isView) {
            float[] valueCopy = new float[this.size];
            valueCopyRec(new int[this.dimensions.length], valueCopy, this.dimensions.length - 1, new int[] {0});
            copy.values = valueCopy;
        } else {
            copy.values = this.values.clone();
        }
        
        copy.size = this.size;
        
        copy.calculateOffsets(dimensions);
        copy.startIndex = 0;

        return copy;
    }

}