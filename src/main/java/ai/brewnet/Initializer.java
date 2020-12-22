package ai.brewnet;

import java.util.Random;
import java.util.function.Supplier;

/**
 * Different strategies to initializing weights in a neural network
 *
 *  Xavier      - Best used for layers that use the tanh activation function
 *  Kaiming     - Best used for layers that use the ReLU activation function
 *  Gaussian    - Not very good to use, maybe you have a use for it. Just use Kaiming or Xavier
 *
 * @author Jordan Florchinger
 */
public abstract class  Initializer {

    public abstract Matrix2D initMatrix(final int rows, final int cols);

    private Matrix2D createDoubleMatrix(final int rows, final int cols, Supplier<Double> doubleSupplier) {
        final Matrix2D matrix2D = new Matrix2D(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix2D.doubles[i][j] = doubleSupplier.get();
            }
        }
        return matrix2D;
    }

    /**
     * 1. Generate a random Gaussian number
     * 2. Apply a tanh operation on it
     * 3. Multiply it by Math.sqrt(6.0 / (rows + cols))
     */
    public static class Xavier extends Initializer {
        @Override
        public Matrix2D initMatrix(int rows, int cols) {
            double srt = Math.sqrt(6.0 / (rows + cols));
            return super.createDoubleMatrix(rows, cols, () -> Math.tanh(new Random().nextGaussian()) * srt);
        }
    }

    /**
     * 1. Generate a random Gaussian number
     */
    public static class Gaussian extends Initializer {
        @Override
        public Matrix2D initMatrix(int rows, int cols) {
            return super.createDoubleMatrix(rows, cols, () -> new Random().nextGaussian());
        }
    }

    /**
     * 1. Generate a random Gaussian number
     * 2. Apply a ReLU operation on it
     * 3. Multiply it by Math.sqrt(2) / Math.sqrt(rows)
     */
    public static class Kaiming extends Initializer {
        @Override
        public Matrix2D initMatrix(final int rows, final int cols) {
            double srt = Math.sqrt(2) / Math.sqrt(rows);
            return super.createDoubleMatrix(rows, cols, () -> Math.max(0, new Random().nextGaussian()) * srt);
        }
    }

}
