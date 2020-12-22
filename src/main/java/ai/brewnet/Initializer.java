package ai.brewnet;

import java.util.Random;

public abstract class Initializer {

    public abstract Matrix2D generate(final int rows, final int cols);

    public static class Xavier extends Initializer {

        @Override
        public Matrix2D generate(final int rows, final int cols) {
            final Matrix2D matrix2D = new Matrix2D(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double rand = new Random().nextDouble();
                    matrix2D.doubles[i][j] = rand;
                }
            }
            return matrix2D;
        }

    }

}
