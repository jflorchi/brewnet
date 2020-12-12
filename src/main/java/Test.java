import ai.brewnet.Matrix2D;
import ai.brewnet.Sequential;
import ai.brewnet.activation.impl.Linear;
import ai.brewnet.activation.impl.Relu;
import ai.brewnet.layers.impl.Dense;

public class Test {

    public static void main(String[] args) {

        final Sequential model = new Sequential();

        model.addLayer(new Dense(2, new Relu()));
        model.addLayer(new Dense(4, new Relu()));
        model.addLayer(new Dense(1, new Linear()));

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
