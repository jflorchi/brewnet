package ai.brewnet;

import java.util.LinkedList;

public class Sequential {

    public Activation activationFunction = new Activation.Linear();
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
            if (cur.inputLayer == null) {
                cur.weights = Matrix2D.createRandom(cur.units, cur.units);
            } else {
                cur.weights = Matrix2D.createRandom(cur.units, cur.inputLayer.units);
            }
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
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < x.length; j++) {
                final Matrix2D in = new Matrix2D(new double[][]{x[j]});
                final Matrix2D out = new Matrix2D(new double[][]{y[j]});
                Matrix2D prediction = this.predict(in.transpose());
                System.out.println(prediction);
                System.out.println();
                this.backPropagation(out, prediction);
            }
        }
    }


    /*
    lastOutputMapped * lastLayer activation derivative mapped onto lastOutput * the derivative of the loss
     */
    public void backPropagation(Matrix2D y, Matrix2D yHat) {
        System.out.println("LOSS: " + this.loss.function(yHat, y));
        System.out.println("DERIV_LOSS: " + this.loss.derivative(yHat, y));
        int i = this.layers.size() - 1;
        while (i >= 1) {
            final Layer layer = this.layers.get(i);
            layer.gradient = this.layers.get(i - 1).lastActivOut.mul(layer.lastDerivOut.transpose()).scale(this.loss.derivative(yHat, y)).transpose();
            yHat = layer.lastActivOut;
            layer.weights = layer.weights.sub(layer.gradient.scale(this.optimizer.learningRate));
            i--;
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
     * @return the output of the current layer to be passed on to next layer
     */
    private Matrix2D forwardPropagation(Matrix2D input) {
        Matrix2D out = new Matrix2D(0, 0);
        for (Layer layer : this.layers) {
            out = layer.weights.mul(input).add(layer.biases);
            layer.lastDerivOut = Activation.mapDerivative(out, layer.activation);
            out = Activation.mapFunction(out, layer.activation);
            layer.lastActivOut = out;
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
