package ai.brewnet.loss;

public abstract class Loss {

    public abstract double compute(double[] output, double[] expected);

}

