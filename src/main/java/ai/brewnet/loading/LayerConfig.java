package io.aviea.ml.loading;

import io.aviea.ml.Tensor;
import io.aviea.ml.Tensor2D;
import io.aviea.ml.activation.impl.Linear;
import io.aviea.ml.activation.impl.Relu;
import io.aviea.ml.activation.impl.Softmax;
import io.aviea.ml.layers.Layer;
import io.aviea.ml.layers.impl.DenseLayer;
import org.ujmp.core.doublematrix.DenseDoubleMatrix2D;

public class LayerConfig {

    private final String type, activation;
    private final int size;
    private Tensor2D weights;
    private Tensor2D biases;

    public LayerConfig(String type, String activation, int size) {
        this.type = type;
        this.activation = activation;
        this.size = size;
    }

    public String getType() {
        return type;
    }

    public int getSize() {
        return size;
    }

    public String getActivation() {
        return activation;
    }

    public void setBiases(Tensor2D biases) {
        this.biases = biases;
    }

    public void setWeights(Tensor2D weights) {
        this.weights = weights;
    }

    /**
     * Builds a layer based on the configuration
     *
     * @return the layer
     */
    public Layer buildLayer() {
        if (this.activation.equals("relu")) {
            return new DenseLayer(this.weights, this.biases, new Relu());
        } else if (this.activation.equals("linear")) {
            return new DenseLayer(this.weights, this.biases, new Linear());
        } else if (this.activation.equals("softmax")) {
            return new DenseLayer(this.weights, this.biases, new Softmax());
        }
        return new DenseLayer(this.weights, this.biases, new Linear());
    }

}
