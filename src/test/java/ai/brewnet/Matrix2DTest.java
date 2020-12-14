package ai.brewnet;

import org.junit.jupiter.api.Test;

public class Matrix2DTest {

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
        final Matrix2D result = new Matrix2D(new double[][]{
                new double[]{29, -16},
                new double[]{38, 6}
        });
        assert t1.mtimes(t2).equals(result);
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
        final Matrix2D result = new Matrix2D(new double[][]{
                new double[]{3},
                new double[]{9}
        });
        assert t1.madd(t2).equals(result);
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
