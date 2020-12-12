package ai.brewnet.loss.impl;

import ai.brewnet.loss.Loss;

public class MeanSquaredAbsoluteError extends Loss {

    @Override
    public double compute(double[] output, double[] expected) {
        double total = 0;
        for (int i = 0; i < output.length; i++) {
            total += Math.abs(output[i] - expected[i]);
        }
        return total / output.length;
    }

}
