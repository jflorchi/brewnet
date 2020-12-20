package ai.brewnet;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class VectorTest {

    @Test
    public void testScaling() {
        final Vector v1 = new Vector(new double[]{0.5, 0.2, 0.3, 0.5});
        final Vector v2 = new Vector(new double[]{0.5, 0.2, 0.3});

        assert v1.scale(1.0).equals(v1);
        assert v1.scale(0.5).equals(new Vector(new double[]{0.25, 0.1, 0.15, 0.25}));
        assert v2.scale(-1.0).equals(new Vector(new double[]{-0.5, -0.2, -0.3}));
    }

    @Test
    public void testMultiply() {
        final Vector v1 = new Vector(new double[]{0.5, 0.2, 0.3, 0.5});
        final Vector v2 = new Vector(new double[]{0.5, 0.2, 0.3, 0.5});
        assert v1.mul(v2) == 0.63;

        final Vector v3 = new Vector(new double[]{0.5, 0.2, 0.3});
        final Vector v4 = new Vector(new double[]{0.5, 0.2, 0.3, 0.5});
        Assertions.assertThrows(IllegalArgumentException.class, () -> {
            v3.mul(v4);
        });

        final Vector v5 = new Vector(new double[]{0.5, 0.2, 0.3, 0.21});
        final Vector v6 = new Vector(new double[]{0.5, 0.3, 0.5});
        Assertions.assertThrows(IllegalArgumentException.class, () -> {
            v5.mul(v6);
        });
    }

    @Test
    public void testAddition() {
        final Vector v1 = new Vector(new double[]{0.5, 0.2, 0.3, 0.5});
        final Vector v2 = new Vector(new double[]{0.5, 0.2, 0.3, 0.5});
        assert v1.add(v2).equals(new Vector(new double[]{1.0, 0.4, 0.6, 1.0}));

        final Vector v3 = new Vector(new double[]{0.5, 0.2, 0.3});
        final Vector v4 = new Vector(new double[]{0.5, 0.2, 0.3, 0.5});
        Assertions.assertThrows(IllegalArgumentException.class, () -> {
            v3.add(v4);
        });

        final Vector v5 = new Vector(new double[]{0.5, 0.2, 0.3, 0.21});
        final Vector v6 = new Vector(new double[]{0.5, 0.3, 0.5});
        Assertions.assertThrows(IllegalArgumentException.class, () -> {
            v5.add(v6);
        });
    }

    @Test
    public void testSubtraction() {
        final Vector v1 = new Vector(new double[]{0.4, 0.5, 0.1, 0.4});
        final Vector v2 = new Vector(new double[]{0.5, 0.2, 0.3, 0.5});
        assert v1.sub(v2).equals(new Vector(new double[]{-0.09999999999999998, 0.3, -0.19999999999999998, -0.09999999999999998}));

        final Vector v3 = new Vector(new double[]{0.5, 0.2, 0.3});
        final Vector v4 = new Vector(new double[]{0.5, 0.2, 0.3, 0.5});
        Assertions.assertThrows(IllegalArgumentException.class, () -> {
            v3.sub(v4);
        });

        final Vector v5 = new Vector(new double[]{0.5, 0.2, 0.3, 0.21});
        final Vector v6 = new Vector(new double[]{0.5, 0.3, 0.5});
        Assertions.assertThrows(IllegalArgumentException.class, () -> {
            v5.sub(v6);
        });
    }

    @Test
    public void testMatrixConversion() {
        final Vector v = new Vector(new double[]{1, 2, 3});
        final Matrix2D m = new Matrix2D(new double[][]{
                new double[]{1},
                new double[]{2},
                new double[]{3},
        });
        assert v.toMatrix().equals(m);
    }

}
