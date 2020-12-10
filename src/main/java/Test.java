import ai.brewnet.Matrix2D;
import ai.brewnet.Sequential;
import ai.brewnet.activation.impl.Linear;
import ai.brewnet.activation.impl.Relu;
import ai.brewnet.layers.impl.Dense;

public class Test {

    public static void main(String[] args) {

        final Sequential model = new Sequential();

        model.addLayer(new Dense(8, new Relu()));
        model.addLayer(new Dense(64, new Relu()));
        model.addLayer(new Dense(3, new Linear()));

        model.compile();

        final double[][] x = new double[][]{
                Matrix2D.createRandom(8),
                Matrix2D.createRandom(8),
                Matrix2D.createRandom(8),
                Matrix2D.createRandom(8),
                Matrix2D.createRandom(8),
                Matrix2D.createRandom(8)
        };
        final double[][] y = new double[][]{
                Matrix2D.createRandom(3),
                Matrix2D.createRandom(3),
                Matrix2D.createRandom(3),
                Matrix2D.createRandom(3),
                Matrix2D.createRandom(3),
                Matrix2D.createRandom(3)
        };

        model.fit(x, y);

    }

}
