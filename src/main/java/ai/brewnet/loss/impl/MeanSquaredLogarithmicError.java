package ai.brewnet.loss.impl;

import ai.brewnet.loss.Loss;

public class MeanSquaredLogarithmicError extends Loss {

    @Override
    public double compute(double[] output, double[] expected) {
        double total = 0;
        for (int i = 0; i < output.length; i++) {
            total += Math.pow(Math.log(output[i] + 1) - Math.log(expected[i] + 1), 2);
        }
        return total / output.length;
    }

}
