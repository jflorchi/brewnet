import ai.brewnet.Activation;
import ai.brewnet.Sequential;
import ai.brewnet.Layer;

public class Test {

    public static void main(String[] args) {

        final Sequential model = new Sequential();

        model.addLayer(new Layer.Dense(2, new Activation.Relu()));
        model.addLayer(new Layer.Dense(4, new Activation.Relu()));
        model.addLayer(new Layer.Dense(1, new Activation.Linear()));

        model.compile();

        final double[][] x = new double[][]{
                new double[]{0, 0},
                new double[]{0, 1},
                new double[]{1, 0},
                new double[]{1, 1},
        };
        final double[][] y = new double[][]{
                new double[]{0},
                new double[]{1},
                new double[]{1},
                new double[]{1},
        };

        model.fit(x, y);

    }

}
