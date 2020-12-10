package ai.brewnet.layers.impl;

import ai.brewnet.activation.ActivationFunction;
import ai.brewnet.layers.Layer;

public class Dense extends Layer {

    public Dense(int units, ActivationFunction activationFunction) {
        this.units = units;
        this.activationFunction = activationFunction;
    }

}
