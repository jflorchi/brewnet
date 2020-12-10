package ai.brewnet.activation.impl;

import ai.brewnet.activation.ActivationFunction;

/**
 * Rectified linear unit
 * <p>
 * f(x) = max(0, x)
 *
 * @author Parametric
 */
public class Relu implements ActivationFunction {


    /**
     * @inheritdoc
     */
    @Override
    public double activate(double x) {
        return Math.max(0, x);
    }


    /**
     * @inheritdoc
     */
    @Override
    public String getIdentifier() {
        return "relu";
    }

}
