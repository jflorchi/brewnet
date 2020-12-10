package ai.brewnet.layers;

import ai.brewnet.Matrix2D;
import ai.brewnet.activation.ActivationFunction;

/**
 * Base class for creating new layers
 * Only used to create Dense right now therefore limited functionality
 *
 * @author Parametric
 */
public abstract class Layer {

    public int units;
    public Matrix2D weights;
    public Matrix2D biases;
    public Layer outputLayer, inputLayer;
    public ActivationFunction activationFunction;

    @Override
    public String toString() {
        return getClass().getName() + "[units=" + this.units
                + ", activation_function=" + this.activationFunction.toString()
                + "]";
    }

}
