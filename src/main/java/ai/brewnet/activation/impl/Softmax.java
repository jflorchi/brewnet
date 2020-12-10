package ai.brewnet.activation.impl;

import ai.brewnet.activation.ActivationFunction;

import java.util.Arrays;

public class Softmax implements ActivationFunction {

    private double[] inputVector;

    public void setInputVector(double[] inputVector) {
        this.inputVector = inputVector;
    }

    @Override
    public double activate(double x) {
        double total = Arrays.stream(inputVector).map(Math::exp).sum();
        return Math.exp(x) / total;
    }

    @Override
    public String getIdentifier() {
        return "softmax";
    }

}
