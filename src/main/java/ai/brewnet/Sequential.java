package ai.brewnet;

import java.util.LinkedList;

public class Sequential {

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
            cur.weights = Matrix2D.createRandom(cur.inputLayer == null ? cur.units : cur.inputLayer.units, cur.units);
            cur.biases = Vector.createZeros(cur.units);
        }
    }

    /**
     * Calls the forward propagation function starting from layer 0 and uses the network inputs
     *
     * @param input network inputs
     * @return the predicted values from the network
     */
    public Matrix2D predict(Matrix2D input) {
        return this.forwardPropagation(input);
    }


    /**
     * Fit the training data to the model. AKA, training the model.
     *
     * The lengths of the two arrays must be the same. For every input, there has to be an output.
     * @param x a 2D double array where each sub array is an input to the network
     * @param y a 2D double array where each sub array is the expected output of the network
     */
    public void fit(final double[][] x, final double[][] y) {
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < x.length; j++) {
                final Matrix2D in = new Matrix2D(new double[][]{x[j]}).transpose();
                final Matrix2D out = new Matrix2D(new double[][]{y[j]}).transpose();
                Matrix2D prediction = this.predict(in);
                System.out.println("LOSS: " + this.loss.function(prediction, out));
                this.backPropagation(out, prediction);
            }
        }
    }



    private void backPropagation(Matrix2D y, Matrix2D yHat) {
        final Layer last = this.layers.getLast();
        Matrix2D delta = this.loss.derivative(yHat, y).hadamard(Activation.mapDerivative(last.prevOutput, last.activation));
        last.weights = last.weights.add(last.inputLayer.lastOutputActivationMapped.mul(delta.transpose().scale(this.optimizer.learningRate)));
        int i = this.layers.size() - 2;
        while (i >= 1) {
            final Layer l = this.layers.get(i);
            final Layer lp1 = this.layers.get(i + 1);
            delta = lp1.weights.mul(delta).hadamard(Activation.mapDerivative(l.prevOutput, l.activation));
            l.weights = l.weights.add(l.inputLayer.lastOutputActivationMapped.mul(delta.transpose().scale(this.optimizer.learningRate)));
            i--;
        }
        // is the first layer weights being updated?
        // something going on here, need to get my indexes right
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
    private Matrix2D forwardPropagation(Matrix2D input) {
        Matrix2D out = new Matrix2D(0, 0);
        for (Layer layer : this.layers) {
            out = layer.weights.transpose().mul(input).add(layer.biases);
            layer.prevOutput = out;
            out = Activation.mapFunction(out, layer.activation);
            layer.lastOutputActivationMapped = out;
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

    public Matrix2D clone() {
        try {
            return (Matrix2D) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }

}
