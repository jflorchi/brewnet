package ai.brewnet.activation;

public interface ActivationFunction {


    /**
     * Applies the activation function onto x and returns y
     *
     * @param x x
     * @return y
     */
    double activate(double x);


    /**
     * Future for resolving activations via functions
     *
     * @return the activation identifier
     */
    String getIdentifier();

}
