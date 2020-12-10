package ai.brewnet;

import ai.brewnet.activation.ActivationFunction;
import ai.brewnet.activation.impl.Linear;
import ai.brewnet.activation.impl.Softmax;
import ai.brewnet.layers.Layer;

import java.util.LinkedList;

public class Sequential {

    private final ActivationFunction activationFunction = new Linear();
    private final LinkedList<Layer> layers = new LinkedList<>();

    public void addLayer(final Layer layer) {
        if (this.layers.size() > 0) {
            final Layer last = this.layers.getLast();
            last.outputLayer = layer;
            layer.inputLayer = last;
        }
        this.layers.add(layer);
    }

    /**
     * Initializes the weights and biases matrices
     */
    public void compile() {
        for (Layer cur : this.layers) {
            if (cur.inputLayer == null) {
                cur.weights = Matrix2D.createRandom(cur.units, cur.units);
            } else if (cur.outputLayer == null) {
                cur.weights = Matrix2D.createRandom(cur.units, cur.units);
                cur.inputLayer.weights = Matrix2D.createRandom(cur.units, cur.inputLayer.inputLayer.units);
            } else {
                cur.weights = Matrix2D.createRandom(cur.units, cur.inputLayer.units);
            }
            cur.biases = Matrix2D.createRandom(cur.units, 1);
        }
    }

    /**
     * Calls the forward propagation function starting from layer 0 and uses the network inputs
     *
     * @param input network inputs
     * @return the predicted values from the network
     */
    public Matrix2D predict(Matrix2D input) {
        return this.forwardPropagation(input, this.layers.getFirst());
    }


    public void fit(final double[][] x, final double[][] y) {
        for (int i = 0; i < x.length; i++) {
            double[] ds = x[i];
            final Matrix2D in = new Matrix2D(new double[][]{ds});
            Matrix2D prediction = this.predict(in.transpose());
            System.out.println(prediction);
        }
    }


    /**
     * Performs the forward propagation algorithm on the provided input and the layers of the network
     * <p>
     * If the layer is null map the network activation function on the input and return
     * <p>
     * 1. Multiply the weights by the input vector
     * 2. Add the biases to the result of the above operation
     * 3. Map the activation function on the result of the above operation
     *
     * @param input      the matrix of values
     * @param firstLayer the next layer that the values will be fed into after mutation
     * @return the output of the current layer to be passed on to next layer
     */
    private Matrix2D forwardPropagation(Matrix2D input, Layer firstLayer) {
        // If we are done the provided layer is null so map the final activation function and return
        if (firstLayer == null) {
            this.map(input, this.activationFunction);
            return input;
        }
//        this.map(firstLayer.weights.multiply(input).add(firstLayer.biases), firstLayer.activationFunction);
        // Multiply the weight matrix by the current value matrix
        Matrix2D output = firstLayer.weights.multiply(input);
        // Add the bias
        output = output.add(firstLayer.biases.transpose());
        // map the layers activation function onto the layer output
        this.map(output, firstLayer.activationFunction);
        // recursive call
        return forwardPropagation(output, firstLayer.outputLayer);
    }


    /**
     * Maps the provided activation function on the provided DoubleMatrix
     * DoubleMatrix is mutated
     *
     * @param matrix             the matrix
     * @param activationFunction the activation function
     */
    private void map(final Matrix2D matrix, final ActivationFunction activationFunction) {
        if (activationFunction instanceof Softmax) {
            final double[] vector = new double[(int) matrix.getRowCount()];
            for (int i = 0; i < matrix.getRowCount(); i++) {
                vector[i] = matrix.doubles[i][0];
            }
            ((Softmax) activationFunction).setInputVector(vector);
        }
        for (int i = 0; i < matrix.getRowCount(); i++) {
            double[] tmp = matrix.doubles[i];
            for (int j = 0; j < matrix.getColumnCount(); j++) {
                tmp[j] = activationFunction.activate(tmp[j]);
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        for (Layer layer : this.layers) {
            str.append(layer.toString()).append("\n");
        }
        return str.toString();
    }
}
