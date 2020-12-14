package ai.brewnet;

public abstract class Loss {

    public abstract double function(double[] output, double[] expected);

    public abstract double derivative(double[] output, double[] expected);

    public static class MeanSquaredLogarithmicError extends Loss {

        @Override
        public double function(double[] output, double[] expected) {
            double total = 0;
            for (int i = 0; i < output.length; i++) {
                total += Math.pow(Math.log(output[i] + 1) - Math.log(expected[i] + 1), 2);
            }
            return total / output.length;
        }

        @Override
        public double derivative(double[] output, double[] expected) {
            return 0;
        }

    }

    public static class MeanSquaredError extends Loss {

        @Override
        public double function(double[] output, double[] expected) {
            double total = 0;
            for (int i = 0; i < output.length; i++) {
                total += Math.pow(expected[i] - output[i], 2) / 2;
            }
            return total / output.length;
        }

        @Override
        public double derivative(double[] output, double[] expected) {
            return 0;
        }

    }

    public static class MeanSquaredAbsoluteError extends Loss {

        @Override
        public double function(double[] output, double[] expected) {
            double total = 0;
            for (int i = 0; i < output.length; i++) {
                total += Math.abs(output[i] - expected[i]);
            }
            return total / output.length;
        }

        @Override
        public double derivative(double[] output, double[] expected) {
            return 0;
        }

    }

}

