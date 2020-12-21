package ai.brewnet;

import java.util.Arrays;

public abstract class Loss {

    public abstract double function(Matrix2D output, Matrix2D expected);

    public abstract Matrix2D derivative(Matrix2D output, Matrix2D expected);

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
            if (!output.shape().equals(expected.shape())) {
                throw new IllegalArgumentException("output shape != expected shape " + output.shape() + " != " + expected.shape());
            }
            double avg = 0;
            for (int row = 0; row < output.doubles.length; row++) {
                double total = 0;
                for (int col = 0; col < output.doubles[row].length; col++) {
                    total += Math.pow(expected.doubles[row][col] - output.doubles[row][col], 2);
                }
                avg += total / output.doubles[row].length;
            }
            return avg / output.doubles.length;
        }

        @Override
        public Matrix2D derivative(Matrix2D output, Matrix2D expected) {
            if (!output.shape().equals(expected.shape())) {
                throw new IllegalArgumentException("output shape != expected shape " + output.shape() + " != " + expected.shape());
            }
            Matrix2D m2 = new Matrix2D(output.getRowCount(), output.getColumnCount());
            for (int row = 0; row < output.doubles.length; row++) {
                for (int col = 0; col < output.doubles[row].length; col++) {
                    m2.doubles[row][col] = 2 * (expected.doubles[row][col] - output.doubles[row][col]);
                }
            }
            return m2;
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

