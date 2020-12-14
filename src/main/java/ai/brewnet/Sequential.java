package ai.brewnet;

import java.util.LinkedList;

public class Sequential {

    public Activation activationFunction = new Activation.Linear();
    public LinkedList<Layer> layers = new LinkedList<>();
    public Loss loss = new Loss.MeanSquaredError();
    public Optimizer optimizer = new Optimizer.SGD();

    public Sequential() {

    }

    public Sequential(final LinkedList<Layer> layers) {
        this.layers = layers;
    }

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
            cur.biases = Matrix2D.createRandomZeros(cur.units, 1);
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


    /**
     * Fit the training data to the model. AKA, training the model.
     *
     * The lengths of the two arrays must be the same. For every input, there has to be an output.
     * @param x a 2D double array where each sub array is an input to the network
     * @param y a 2D double array where each sub array is the expected output of the network
     */
    public void fit(final double[][] x, final double[][] y) {
        for (int i = 0; i < x.length; i++) {
            final Matrix2D in = new Matrix2D(new double[][]{x[i]});
            final Matrix2D out = new Matrix2D(new double[][]{y[i]});
            Matrix2D prediction = this.predict(in.transpose()).transpose();
            this.backPropagation(in, out, prediction, this.layers.getLast());
        }
    }


    public void backPropagation(final Matrix2D x, Matrix2D y, Matrix2D yHat, final Layer lastLayer) {
        final double loss = this.loss.function(yHat.doubles[0], y.doubles[0]);
        final Matrix2D wGrad = this.optimizer.weightGradient(lastLayer.weights, lastLayer.gradients, y, yHat, loss);
        lastLayer.weights = lastLayer.weights.sub(this.optimizer.learningRate).mtimes(wGrad);

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
        // Multiply the weight matrix by the current value matrix and add the bias
        Matrix2D output = firstLayer.weights.mtimes(input).madd(firstLayer.biases.transpose());
        // apply the activation function
        this.map(output, firstLayer.activation);
        firstLayer.gradients = output;
        // recurse onto next layer
        return forwardPropagation(output, firstLayer.outputLayer);
    }


    /**
     * Maps the provided activation function on the provided DoubleMatrix
     * DoubleMatrix is mutated
     *
     * @param matrix             the matrix
     * @param activation the activation function
     */
    private void map(final Matrix2D matrix, final Activation activation) {
        if (activation instanceof Activation.Softmax) {
            final double[] vector = new double[(int) matrix.getRowCount()];
            for (int i = 0; i < matrix.getRowCount(); i++) {
                vector[i] = matrix.doubles[i][0];
            }
            ((Activation.Softmax) activation).inputVector = vector;
        }
        for (int i = 0; i < matrix.getRowCount(); i++) {
            double[] tmp = matrix.doubles[i];
            for (int j = 0; j < matrix.getColumnCount(); j++) {
                tmp[j] = activation.activate(tmp[j]);
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
