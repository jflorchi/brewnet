package ai.brewnet;

import java.util.Arrays;

/**
 * Provides Activation functions for the neural network implementation.
 * They are all stored in one class file to shrink source code size
 *
 * @author Jordan Florchinger
 */
public abstract class Activation {

    /**
     * Maps the activation function onto a Matrix2D. This does not mutate the provided matrix.
     * The matrix is cloned before the function is applied and then returned.
     * @param matrix2D      the Matrix2D to clone and map onto
     * @param activation    the activation function to map onto the cloned Matrix2D
     * @return              the matrix that is mapped by the function
     */
    public static Matrix2D mapFunction(final Matrix2D matrix2D, final Activation activation) {
        final Matrix2D m2 = new Matrix2D(matrix2D);
        if (activation instanceof Activation.Softmax) {
            final double[] vector = new double[(int) m2.getRowCount()];
            for (int i = 0; i < m2.getRowCount(); i++) {
                vector[i] = m2.doubles[i][0];
            }
            ((Activation.Softmax) activation).inputVector = vector;
        }
        for (int i = 0; i < m2.getRowCount(); i++) {
            double[] tmp = m2.doubles[i];
            for (int j = 0; j < m2.getColumnCount(); j++) {
                tmp[j] = activation.activate(tmp[j]);
            }
        }
        return m2;
    }

    /**
     * Maps the activation function's derivative onto a Matrix2D. This does not mutate the provided matrix.
     * The matrix is cloned before the derivative is applied and then returned.
     * @param matrix2D      the Matrix2D to clone and map onto
     * @param activation    the activation function's derivative to map onto the cloned Matrix2D
     * @return              the matrix that is mapped by the derivative
     */
    public static Matrix2D mapDerivative(final Matrix2D matrix2D, final Activation activation) {
        final Matrix2D m2 = new Matrix2D(matrix2D);
        for (int i = 0; i < m2.getRowCount(); i++) {
            double[] tmp = m2.doubles[i];
            for (int j = 0; j < m2.getColumnCount(); j++) {
                tmp[j] = activation.derivative(tmp[j]);
            }
        }
        return m2;
    }

    /**
     * Applies the activation function onto x and returns y
     *
     * @param x x
     * @return y
     */
    public abstract double activate(double x);

    /**
     * Applies the activation function derivative onto x and returns y
     *
     * @param x x
     * @return y
     */
    public abstract double derivative(double x);

    /**
     * Rectified linear unit
     * f(x) = max(0, x)
     */
    public static class Relu extends Activation {

        @Override
        public double activate(double x) {
            return Math.max(0, x);
        }

        @Override
        public double derivative(double x) {
            return x >= 0 ? 1D : 0D;
        }

    }

    /**
     * Linear Activation Function f(x) = x
     */
    public static class Linear extends Activation {

        @Override
        public double activate(double x) {
            return x;
        }

        @Override
        public double derivative(double x) {
            return 1;
        }

    }

    /**
     * Softmax, scales all values in the input vector such that the sum of all values == 1
     * This is used for classification because then you get a probability distribution of the classes
     */
    public static class Softmax extends Activation {

        public double[] inputVector = new double[0];

        @Override
        public double activate(double x) {
            double total = Arrays.stream(inputVector).map(val -> val * val).sum();
            return Math.exp(x) / total;
        }

        @Override
        public double derivative(double x) {
            double total = Arrays.stream(inputVector).map(val -> val * 2.0).sum();
            return Math.exp(x) / total;
        }

    }

}
