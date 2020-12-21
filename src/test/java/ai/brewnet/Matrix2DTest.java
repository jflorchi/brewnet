package ai.brewnet;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class Matrix2DTest {

    @Test
    public void testHadamard() {
        final Matrix2D t1 = new Matrix2D(new double[][]{
                new double[]{0, 3, 5},
                new double[]{5, 5, 2}
        });
        final Matrix2D t2 = new Matrix2D(new double[][]{
                new double[]{0, 3, 5},
                new double[]{5, 5, 2}
        });
        final Matrix2D t3 = new Matrix2D(new double[][]{
                new double[]{0.0, 9.0, 25.0},
                new double[]{25.0, 25.0, 4.0},
        });
        final Matrix2D t4 = new Matrix2D(new double[][]{
                new double[]{0.0, 9.0, 25.0},
                new double[]{25.0, 25.0, 4.0},
                new double[]{25.0, 25.0, 4.0},
        });
        assert t1.hadamard(t2).equals(t3);
        Assertions.assertThrows(IllegalArgumentException.class, () -> t1.hadamard(t4));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t2.hadamard(t4));
    }

    @Test
    public void testMultiply() {
        final Matrix2D t1 = new Matrix2D(new double[][]{
                new double[]{0, 3, 5},
                new double[]{5, 5, 2}
        });
        final Matrix2D t2 = new Matrix2D(new double[][]{
                new double[]{3, 4},
                new double[]{3, -2},
                new double[]{4, -2}
        });
        final Matrix2D t3 = new Matrix2D(new double[][]{
                new double[]{29, -16},
                new double[]{38, 6}
        });
        final Matrix2D t4 = new Matrix2D(new double[][]{
                new double[]{20.0, 29.0, 23.0},
                new double[]{-10.0, -1.0, 11.0},
                new double[]{-10.0, 2.0, 16.0},
        });
        assert t1.mul(t2).equals(t3);
        assert t2.mul(t1).equals(t4);
        assert t3.mul(t1).equals(new Matrix2D(new double[][]{
                new double[]{-80.0, 7.0, 113.0},
                new double[]{30.0, 144.0, 202.0}
        }));
        assert t2.mul(t3).equals(new Matrix2D(new double[][]{
                new double[]{239.0, -24.0},
                new double[]{11.0, -60.0},
                new double[]{40.0, -76.0}
        }));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t1.mul(t3));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t3.mul(t2));
    }

    @Test
    public void testAdd() {
        final Matrix2D t1 = new Matrix2D(new double[][]{
                new double[]{0},
                new double[]{5}
        });
        final Matrix2D t2 = new Matrix2D(new double[][]{
                new double[]{3, 4}
        });
        final Matrix2D t3 = new Matrix2D(new double[][]{
                new double[]{3},
                new double[]{9}
        });
        final Matrix2D t4 = new Matrix2D(new double[][]{
                new double[]{4},
                new double[]{10}
        });
        Assertions.assertThrows(IllegalArgumentException.class, () -> t1.add(t2));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t2.add(t1));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t2.add(t3));
        assert t2.add(new Vector(t1.doubles[0])).equals(t2);
        assert t1.add(new Vector(t2.doubles[0])).equals(t3);
        assert t3.add(1).equals(t4);
    }

    @Test
    public void testSub() {
        final Matrix2D t1 = new Matrix2D(new double[][]{
                new double[]{0},
                new double[]{5}
        });
        final Matrix2D t2 = new Matrix2D(new double[][]{
                new double[]{3, 4}
        });
        final Matrix2D t3 = new Matrix2D(new double[][]{
                new double[]{3},
                new double[]{9}
        });
        final Matrix2D t4 = new Matrix2D(new double[][]{
                new double[]{4},
                new double[]{10}
        });
        Assertions.assertThrows(IllegalArgumentException.class, () -> t1.sub(t2));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t2.sub(t1));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t2.sub(t3));
        assert t2.sub(new Vector(t1.doubles[0])).equals(t2);
        assert t3.sub(new Vector(t2.doubles[0])).equals(t1);
        assert t4.sub(1).equals(t3);
    }

    @Test
    public void testScale() {
        final Matrix2D t1 = new Matrix2D(new double[][]{
                new double[]{1, 2, 3},
                new double[]{4, 5, 6}
        });
        final Matrix2D t2 = new Matrix2D(new double[][]{
                new double[]{2, 4, 6},
                new double[]{8, 10, 12}
        });
        final Matrix2D t3 = new Matrix2D(new double[][]{
                new double[]{-2, -4, -6},
                new double[]{-8, -10, -12}
        });
        assert t1.scale(2.0).equals(t2);
        assert t1.scale(-2.0).equals(t3);
    }

    @Test
    public void testDiv() {
        final Matrix2D t1 = new Matrix2D(new double[][]{
                new double[]{1, 2, 3},
                new double[]{4, 5, 6}
        });
        final Matrix2D t2 = new Matrix2D(new double[][]{
                new double[]{2, 4, 6},
                new double[]{8, 10, 12}
        });
        final Matrix2D t3 = new Matrix2D(new double[][]{
                new double[]{-1, -2, -3},
                new double[]{-4, -5, -6}
        });
        assert t2.div(2).equals(t1);
        assert t2.div(-2).equals(t3);
    }

    @Test
    public void testTranspose() {
        final Matrix2D t1 = new Matrix2D(new double[][]{
                new double[]{0, 3, 6},
                new double[]{5, 6, 1}
        });
        final Matrix2D result = new Matrix2D(new double[][]{
                new double[]{0, 5},
                new double[]{3, 6},
                new double[]{6, 1}
        });
        assert t1.transpose().equals(result);
    }

}
