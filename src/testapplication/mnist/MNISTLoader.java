package testapplication.mnist;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import tensors.Vector;

/**
 *
 * @author bierdi
 */
public class MNISTLoader {
    
    private static final int IMAGE_SIZE = 28 * 28;
    private static final float MAX_PIXEL_VALUE = 255.0f;
    
    public static List<LabelImagePair> getData(File file) throws FileNotFoundException {
        
        List<LabelImagePair> data = new ArrayList<>();
        Scanner scanner = new Scanner(file);
        
        while(scanner.hasNext()) {
        
            String line = scanner.nextLine();
            String[] cut = line.split(",");
            
            int label = Integer.parseInt(cut[0]);
            Vector image = new Vector(IMAGE_SIZE);

            for (int i = 0; i < IMAGE_SIZE; i++) {
                image.setValueFlat(Integer.parseInt(cut[i + 1]) / MAX_PIXEL_VALUE * 2 - 1, i);
            }
         
            data.add(new LabelImagePair(label, image));
            
        }
        
        return data;
    }
    
}
