
package testapplication.mnist;

import tensors.Vector;

/**
 *
 * @author bierdi
 */
public record LabelImagePair(int label, Vector image) {}
