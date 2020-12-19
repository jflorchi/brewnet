package ai.brewnet;

public abstract class Optimizer {

    public double learningRate = 0.0001;

    public abstract Matrix2D weightGradient(final Matrix2D w, final Matrix2D in, final Matrix2D y,
                                            final Matrix2D yHat, final double loss);

    public abstract Matrix2D biasGradient(final Matrix2D w, final Matrix2D in, final Matrix2D y,
                                          final Matrix2D yHat, final double loss);

    public static class SGD extends Optimizer {

        public SGD(final double learningRate) {
            this.learningRate = learningRate;
        }

        /*
            lastOutputMapped * lastLayer activation derivative mapped onto lastOutput * the derivative of the loss
         */
        @Override
        public Matrix2D weightGradient(Matrix2D w, Matrix2D in, Matrix2D y, Matrix2D yHat, double loss) {
            return in.mtimes(yHat.msub(y).transpose()).times((1 / (double) w.doubles[0].length));
        }

        @Override
        public Matrix2D biasGradient(Matrix2D w, Matrix2D in, Matrix2D y, Matrix2D yHat, double loss) {
            return yHat.msub(y).times((1 / (double) w.doubles[0].length));
        }

    }

}
