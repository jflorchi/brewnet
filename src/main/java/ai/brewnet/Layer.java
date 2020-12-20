package ai.brewnet;

/**
 * Base class for creating new layers
 * Only used to create Dense right now therefore limited functionality
 *
 * @author Parametric
 */
public abstract class Layer {

    public int units;
    public Matrix2D weights, gradient, lastActivOut, lastDerivOut;
    public Vector biases;
    public Layer outputLayer, inputLayer;
    public Activation activation;

    public static class Dense extends Layer {

        public Dense(int units, Activation activationFunction) {
            this.units = units;
            this.activation = activationFunction;
        }

        public Dense(Matrix2D weights, Vector biases, Activation function) {
            this.weights = weights;
            this.biases = biases;
            this.activation = function;
        }

    }

    @Override
    public String toString() {
        return getClass().getName() + "[units=" + this.units
                + ", activation_function=" + this.activation.toString()
                + "] = {\n    Weights: " + this.weights.shape() + "\n    Biases: " + this.biases.doubles.length + "\n}";
    }

}
