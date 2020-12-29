package ai.brewnet;

import java.util.Arrays;
import java.util.LinkedList;

public class NeuralNetwork {

    public LinkedList<Layer> layers = new LinkedList<>();
    public Loss loss = new Loss.MeanSquaredError();
    public Optimizer optimizer;

    public void addLayer(final Layer layer) {
        if (this.layers.size() > 0) {
            final Layer last = this.layers.getLast();
            last.outputLayer = layer;
            layer.inputLayer = last;
        }
        this.layers.add(layer);
    }

    /**
     * Create and initialize the networks weight and bias matrices to random values
     */
    public void compile(final Optimizer optimizer) {
        this.optimizer = optimizer;
        for (Layer cur : this.layers) {
            cur.init();
        }
    }

    /**
     * Calls the forward propagation function starting from layer 0 and uses the network inputs
     *
     * @param input network inputs
     * @return the predicted values from the network
     */
    public Matrix2D predict(double[][] input) {
        return this.forward(new Matrix2D(input).transpose());
    }

    /**
     * Fit the training data to the model. AKA, training the model.
     *
     * The lengths of the two arrays must be the same. For every input, there has to be an output.
     * @param x a 2D double array where each sub array is an input to the network
     * @param y a 2D double array where each sub array is the expected output of the network
     */
    public void fit(final double[][] x, final double[][] y, final int epochs, final int batchSize) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("Miss matched training data x.length != y.length " + x.length + " != " + y.length);
        }
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < x.length; j += batchSize) {
                final int size = Math.min(batchSize, x.length - j);
                final double[][] xs = new double[size][];
                final double[][] ys = new double[size][];
                xs[j] = x[i * j];
                ys[j] = x[i * j];
                System.out.println(Arrays.toString(xs));
                final Matrix2D in = new Matrix2D(xs).transpose();
                final Matrix2D out = new Matrix2D(ys).transpose();
                Matrix2D prediction = this.forward(in);
                System.out.println("LOSS: " + this.loss.function(prediction, out));
                this.backward(in, out, prediction);
            }
        }
    }


    /**
     * Propagate the error back through the network and update the weights
     * @param x     the input to the network
     * @param y     the expected output of the network
     * @param yHat  the actual output of the network
     */
    private void backward(Matrix2D x, Matrix2D y, Matrix2D yHat) {
        // Last Layer
        final Layer last = this.layers.getLast();
        Matrix2D delta = this.loss.derivative(yHat, y).hadamard(Activation.mapDerivative(last.prevOutput, last.activation));
        Matrix2D scaledDelta = delta.scale(this.optimizer.learningRate);
        last.weights = last.weights.add(last.inputLayer.lastOutputActivationMapped.mul(scaledDelta.transpose()));
        last.biases = last.biases.add(scaledDelta.averageRows());

        // Hidden Layers
        int i = this.layers.size() - 2;
        while (i >= 1) {
            final Layer l = this.layers.get(i);
            delta = this.apply(delta, l, l.inputLayer.lastOutputActivationMapped);
            i--;
        }

        // First layer - use x as the input layer
        this.apply(delta, this.layers.getFirst(), x);
    }

    /**
     * Function extraction for layer back propagation
     * @param delta gradient matrix
     * @param l     layer to update weights on
     * @param input layer / input that feeds into the layer that we are updating
     * @return      the new gradient matrix
     */
    private Matrix2D apply(Matrix2D delta, Layer l, Matrix2D input) {
        final Layer lp1 = l.outputLayer;
        delta = lp1.weights.mul(delta).hadamard(Activation.mapDerivative(l.prevOutput, l.activation));
        Matrix2D scaledDelta = delta.scale(this.optimizer.learningRate);
        l.weights = l.weights.add(input.mul(scaledDelta.transpose()));
        l.biases = l.biases.add(scaledDelta.averageRows());
        return delta;
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
     * @return the output of the current layer to be passed on to next layer
     */
    private Matrix2D forward(Matrix2D input) {
        Matrix2D out = new Matrix2D(0, 0);
        for (Layer layer : this.layers) {
            out = layer.weights.transpose().mul(input).add(layer.biases);
            layer.prevOutput = new Matrix2D(out);
            out = Activation.mapFunction(out, layer.activation);
            layer.lastOutputActivationMapped = new Matrix2D(out);
            input = out;
        }
        return out;
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
