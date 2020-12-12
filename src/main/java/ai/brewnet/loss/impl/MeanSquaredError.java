package ai.brewnet.loss.impl;

import ai.brewnet.loss.Loss;

public class MeanSquaredError extends Loss {

    @Override
    public double compute(double[] output, double[] expected) {
        double total = 0;
        for (int i = 0; i < output.length; i++) {
            total += Math.pow(expected[i] - output[i], 2) / 2;
        }
        return total / output.length;
    }

}
