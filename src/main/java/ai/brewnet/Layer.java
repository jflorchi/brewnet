package ai.brewnet;

/**
 * Base class for creating new layers
 * Only used to create Dense right now therefore limited functionality
 *
 * @author Parametric
 */
public abstract class Layer {

    public int units;
    public Matrix2D weights, lastOutputActivationMapped, prevOutput;
    public Vector biases;
    public Layer outputLayer, inputLayer;
    public Activation activation;
    public Initializer initializer = new Initializer.Kaiming();

    public void init() {
        this.weights = this.initializer.initMatrix(this.inputLayer == null ? this.units : this.inputLayer.units, this.units);
        this.biases = Vector.createZeros(this.units);
    }

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

        public Dense(Matrix2D weights, Vector biases, Activation function, Initializer initializer) {
            this.weights = weights;
            this.biases = biases;
            this.activation = function;
            this.initializer = initializer;
        }

    }

    @Override
    public String toString() {
        return getClass().getName() + "[units=" + this.units
                + ", activation_function=" + this.activation.toString()
                + "] = {\n    Weights: " + this.weights.shape() + "\n    Biases: " + this.biases.doubles.length + "\n}";
    }

}
