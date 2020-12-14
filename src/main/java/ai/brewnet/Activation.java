package ai.brewnet;

import java.util.Arrays;

public abstract class Activation {

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
     * <p>
     * f(x) = max(0, x)
     *
     * @author Parametric
     */
    public static class Relu extends Activation {

        /**
         * @inheritdoc
         */
        @Override
        public double activate(double x) {
            return Math.max(0, x);
        }

        /**
         * @inheritdoc
         */
        @Override
        public double derivative(double x) {
            return x >= 0 ? 1D : 0D;
        }

    }

    /**
     * Linear Activation Function f(x) = x
     *
     * @author Parametric
     */
    public static class Linear extends Activation {

        /**
         * @inheritdoc
         */
        @Override
        public double activate(double x) {
            return x;
        }

        /**
         * @inheritdoc
         */
        @Override
        public double derivative(double x) {
            return 1;
        }

    }

    public static class Softmax extends Activation {

        public double[] inputVector = new double[0];

        /**
         * @inheritdoc
         */
        @Override
        public double activate(double x) {
            double total = Arrays.stream(inputVector).map(Math::exp).sum();
            return Math.exp(x) / total;
        }

        /**
         * @inheritdoc
         */
        @Override
        public double derivative(double x) {
            return 0;
        }

    }

}
