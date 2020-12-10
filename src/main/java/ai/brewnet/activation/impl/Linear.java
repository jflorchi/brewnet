package ai.brewnet.activation.impl;

import ai.brewnet.activation.ActivationFunction;

/**
 * Linear Activation Function f(x) = x
 *
 * @author Parametric
 */
public class Linear implements ActivationFunction {
    /**
     * @inheritdoc
     */
    @Override
    public double activate(double x) {
        return x;
    }


    /**
     * @inheritdoc
     */
    @Override
    public String getIdentifier() {
        return "linear";
    }

}
