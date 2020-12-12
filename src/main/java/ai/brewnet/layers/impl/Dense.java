package ai.brewnet.layers.impl;

import ai.brewnet.Matrix2D;
import ai.brewnet.activation.ActivationFunction;
import ai.brewnet.layers.Layer;

public class Dense extends Layer {

    public Dense(int units, ActivationFunction activationFunction) {
        this.units = units;
        this.activationFunction = activationFunction;
    }

    public Dense(Matrix2D weights, Matrix2D biases, ActivationFunction function) {
        this.weights = weights;
        this.biases = biases;
        this.activationFunction = function;
    }

}
