package ai.brewnet;

import java.util.Arrays;

public abstract class Loss {

    public abstract double function(Matrix2D output, Matrix2D expected);

    public abstract double derivative(Matrix2D output, Matrix2D expected);

//    public static class MeanSquaredLogarithmicError extends Loss {
//
//        @Override
//        public double function(Matrix2D output, Matrix2D expected) {
//            double total = 0;
//            for (int i = 0; i < output.length; i++) {
//                total += Math.pow(Math.log(output[i] + 1) - Math.log(expected[i] + 1), 2);
//            }
//            return total / output.length;
//        }
//
//        @Override
//        public double derivative(Matrix2D output, Matrix2D expected) {
//            return 1;
//        }
//
//    }

    public static class MeanSquaredError extends Loss {

        @Override
        public double function(Matrix2D output, Matrix2D expected) {
            double avg = 0;
            for (int j = 0; j < output.doubles.length; j++) {
                double total = 0;
                for (int i = 0; i < output.doubles[j].length; i++) {
                    total += Math.pow(expected.doubles[i][j] - output.doubles[j][i], 2);
                }
                avg += total / output.doubles[j].length;
            }
            return avg / output.doubles.length;
        }

        @Override
        public double derivative(Matrix2D output, Matrix2D expected) {
            double avg = 0;
            for (int j = 0; j < output.doubles.length; j++) {
                double total = 0;
                for (int i = 0; i < output.doubles[j].length; i++) {
                    total += 2 * (expected.doubles[i][j] - output.doubles[j][i]);
                }
                avg += total / output.doubles[j].length;
            }
            return avg / output.doubles.length;
        }

    }

//    public static class MeanSquaredAbsoluteError extends Loss {
//
//        @Override
//        public double function(Matrix2D output, Matrix2D expected) {
//            double total = 0;
//            for (int i = 0; i < output.length; i++) {
//                total += Math.abs(output[i] - expected[i]);
//            }
//            return total / output.length;
//        }
//
//        @Override
//        public double derivative(Matrix2D output, Matrix2D expected) {
//            return 1;
//        }
//
//    }

}

