package ai.brewnet;

import java.util.Arrays;
import java.util.Random;

public class Matrix2D {

    public final double[][] doubles;

    public Matrix2D(final double[][] contents) {
        this.doubles = contents;
    }

    public Matrix2D(final int rows, final int cols) {
        this.doubles = new double[rows][cols];
    }

    public Matrix2D(final Matrix2D matrix2D) {
        this.doubles = new double[matrix2D.getRowCount()][matrix2D.getColumnCount()];
        for (int r = 0; r < this.doubles.length; r++) {
            this.doubles[r] = matrix2D.doubles[r].clone();
        }
    }

    public static Matrix2D createRandom(int rows, int cols) {
        final Matrix2D matrix2D = new Matrix2D(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix2D.doubles[i][j] = new Random().nextDouble();
            }
        }
        return matrix2D;
    }

    public static Matrix2D createRandomZeros(int rows, int cols) {
        final Matrix2D matrix2D = new Matrix2D(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix2D.doubles[i][j] = 0;
            }
        }
        return matrix2D;
    }

    public static double[] createRandom(int length) {
        final double[] dubs = new double[length];
        for (int i = 0; i < length; i++) {
            dubs[i] = new Random().nextGaussian();
        }
        return dubs;
    }

    public Matrix2D mtimes(final Matrix2D matrix) {
        final int m1RowCount = this.getRowCount();
        final int m1ColumnCount = this.getColumnCount();
        final int m2RowCount = matrix.getRowCount();
        final int m2ColumnCount = matrix.getColumnCount();
        if (m1ColumnCount != m2RowCount) {
            throw new IllegalArgumentException("a.cols != b.rows: (" + m1ColumnCount + " != " + m2RowCount + ")");
        }
        final Matrix2D rTensor = new Matrix2D(this.getRowCount(), matrix.getColumnCount());
        for (int i = 0; i < m1RowCount; i++) {
            for (int k = 0; k < m2RowCount; k++) {
                double val = this.doubles[i][k];
                for (int j = 0; j < m2ColumnCount; j++) {
                    rTensor.doubles[i][j] += val * matrix.doubles[k][j];
                }
            }
        }
        return rTensor;
    }

    public Matrix2D madd(Matrix2D tensor) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        int m2Rows = tensor.getRowCount();
        if (m1Cols != m2Rows) {
            throw new IllegalArgumentException("a.cols != b.rows: (" + m1Cols + " != " + m2Rows + ")");
        }
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int i = 0; i < m1Rows; i++) {
            matrix2D.doubles[i][0] = this.doubles[i][0] + tensor.doubles[0][i];
        }
        return matrix2D;
    }

    public Matrix2D msub(Matrix2D matrix) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        int m2Rows = matrix.getRowCount();
        if (m1Cols != m2Rows) {
            throw new IllegalArgumentException("a.cols != b.rows: (" + m1Cols + " != " + m2Rows + ")");
        }
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int i = 0; i < m1Cols; i++) {
            System.out.println(this.doubles[0][i] + " - " + matrix.doubles[i][0]);
            matrix2D.doubles[0][i] = this.doubles[0][i] - matrix.doubles[i][0];
        }
        return matrix2D;
    }

    public Matrix2D sub(double val) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int i = 0; i < m1Rows; i++) {
            matrix2D.doubles[i][0] = this.doubles[i][0] - val;
        }
        return matrix2D;
    }

    public Matrix2D times(double val) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int i = 0; i < m1Rows; i++) {
            matrix2D.doubles[i][0] = this.doubles[i][0] * val;
        }
        return matrix2D;
    }

    public Matrix2D transpose() {
        final Matrix2D result = new Matrix2D(this.getColumnCount(), this.getRowCount());
        for (int i = 0; i < this.getRowCount(); i++) {
            for (int j = 0; j < this.getColumnCount(); j++) {
                result.doubles[j][i] = this.doubles[i][j];
            }
        }
        return result;
    }

    public Matrix2D mapActivationFunction(final Activation activation) {
        final Matrix2D m2 = new Matrix2D(this);
//        if (activation instanceof Activation.Softmax) {
//            final double[] vector = new double[(int) m2.getRowCount()];
//            for (int i = 0; i < m2.getRowCount(); i++) {
//                vector[i] = m2.doubles[i][0];
//            }
//            ((Activation.Softmax) activation).inputVector = vector;
//        }
        for (int i = 0; i < m2.getRowCount(); i++) {
            double[] tmp = m2.doubles[i];
            for (int j = 0; j < m2.getColumnCount(); j++) {
                tmp[j] = activation.activate(tmp[j]);
            }
        }
        return m2;
    }

    public Matrix2D mapActivationDerivative(final Activation activation) {
        final Matrix2D m2 = new Matrix2D(this);
        for (int i = 0; i < m2.getRowCount(); i++) {
            double[] tmp = m2.doubles[i];
            for (int j = 0; j < m2.getColumnCount(); j++) {
                tmp[j] = activation.derivative(tmp[j]);
            }
        }
        return m2;
    }

    public int[] shape() {
        return new int[]{this.getRowCount(), this.getColumnCount()};
    }

    public int getRowCount() {
        return this.doubles.length;
    }

    public int getColumnCount() {
        return this.doubles[0].length;
    }

    @Override
    public String toString() {
        final StringBuilder builder = new StringBuilder("[");
        for (double[] ds : this.doubles) {
            builder.append(Arrays.toString(ds)).append(",\n ");
        }
        return builder.substring(0, builder.length() - 3) + "]";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Matrix2D matrix2D = (Matrix2D) o;
        if (matrix2D.doubles.length != this.doubles.length
                || matrix2D.doubles[0].length != this.doubles[0].length) {
            return false;
        }
        double[][] dds = this.doubles;
        for (int i = 0; i < dds.length; i++) {
            double[] ds = dds[i];
            if (!Arrays.equals(ds, matrix2D.doubles[i])) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(doubles);
    }

}
