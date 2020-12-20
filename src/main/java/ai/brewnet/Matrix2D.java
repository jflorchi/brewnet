package ai.brewnet;

import java.awt.*;
import java.util.Arrays;
import java.util.Random;

public class Matrix2D {

    public final double[][] doubles;

    // [rows][cols]
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

    /**
     * Converts an N x M matrix to M x N and moves the values to their new location.
     * In the future, we need to use a transpose boolean target and an accessor like getValue(row, col) which we can then
     * dictate where we're getting the value from. If the transpose flag is enabled we just call doubles[col][row],
     * if not it will call doubles[row][col]. This removes the need to actually move data, it just changes the indexes.
     *
     * @return  the new transposed matrix
     */
    public Matrix2D transpose() {
        final Matrix2D result = new Matrix2D(this.getColumnCount(), this.getRowCount());
        for (int i = 0; i < this.getRowCount(); i++) {
            for (int j = 0; j < this.getColumnCount(); j++) {
                result.doubles[j][i] = this.doubles[i][j];
            }
        }
        return result;
    }

    /**
     * Clones the Matrix2D and then multiplies all values in the matrix by the scalar
     * @param val   the scalar
     * @return      the scaled matrix
     */
    public Matrix2D scale(double val) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int rows = 0; rows < m1Rows; rows++) {
            for (int cols = 0; cols < m1Cols; cols++) {
                matrix2D.doubles[rows][cols] = this.doubles[rows][cols] * val;
            }
        }
        return matrix2D;
    }

    /**
     * Clones the Matrix2D and then divides all values in the matrix by the val
     * @param val   the scalar
     * @return      the scaled matrix
     */
    public Matrix2D div(double val) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int rows = 0; rows < m1Rows; rows++) {
            for (int cols = 0; cols < m1Cols; cols++) {
                matrix2D.doubles[rows][cols] = this.doubles[rows][cols] / val;
            }
        }
        return matrix2D;
    }

    /**
     * Creates a Matrix of the given size and populates it with random double values from 0 to 1
     * @param rows  the number of rows in the matrix
     * @param cols  the number of columsn in the matrix
     * @return      the matrix with populated values
     */
    public static Matrix2D createRandom(int rows, int cols) {
        final Matrix2D matrix2D = new Matrix2D(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix2D.doubles[i][j] = new Random().nextDouble();
            }
        }
        return matrix2D;
    }

    /**
     * Creates a Matrix of the given size and populates it with 0
     * @param rows  the number of rows in the matrix
     * @param cols  the number of columsn in the matrix
     * @return      the matrix with populated zeros
     */
    public static Matrix2D createZeros(int rows, int cols) {
        final Matrix2D matrix2D = new Matrix2D(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix2D.doubles[i][j] = 0;
            }
        }
        return matrix2D;
    }

    /**
     * Clones this matrix and then multiplies it by the provided matrix
     * @param matrix    the provided matrix
     * @return          a new matrix
     */
    public Matrix2D mul(final Matrix2D matrix) {
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

    /**
     * The vector is assumed to be vertical in this context so that the Matrix multiplication is valid
     * The matrix will always be a Nx1 matrix where N = this.getRowCount()
     * @param vector    vector to multiply this by
     * @return          the result matrix
     */
    public Matrix2D mul(final Vector vector) {
        final int m1RowCount = this.getRowCount();
        final int m1ColumnCount = this.getColumnCount();
        if (m1ColumnCount != vector.doubles.length) {
            throw new IllegalArgumentException("a.cols != b.rows: (" + m1ColumnCount + " != " + vector.doubles.length + ")");
        }
        final Matrix2D matrix2D = new Matrix2D(m1RowCount, 1);
        for (int i = 0; i < m1RowCount; i++) {
            for (int k = 0; k < vector.doubles.length; k++) {
                matrix2D.doubles[i][0] += this.doubles[i][k] * vector.doubles[k];
            }
        }
        return matrix2D;
    }

    /**
     * Clones this matrix and the adds the provided matrix to it
     * @param m the matrix to add to the cloned matrix
     * @return  the new matrix
     */
    public Matrix2D add(Matrix2D m) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        int m2Rows = m.getRowCount();
        int m2Cols = m.getColumnCount();
        if (m1Rows != m2Rows) {
            throw new IllegalArgumentException("m1.rows != m2.rows " + m1Rows + " != " + m2Rows);
        } else if (m1Cols != m2Cols) {
            throw new IllegalArgumentException("m1.cols != m2.cols " + m1Cols + " != " + m2Cols);
        }
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int i = 0; i < m1Rows; i++) {
            for (int j = 0; j < m1Cols; j++) {
                matrix2D.doubles[i][j] = this.doubles[i][j] + m.doubles[i][j];
            }
        }
        return matrix2D;
    }

    /**
     * Adds a vector to the matrix, the vector is treated as a column vector
     * @param vector    ...
     * @return          the resulting Matrix2D
     */
    public Matrix2D add(Vector vector) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        if (m1Rows != vector.doubles.length) {
            throw new IllegalArgumentException("a.rows != b.length: (" + m1Rows + " != " + vector.doubles.length + ")");
        }
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int i = 0; i < m1Rows; i++) {
            for (int j = 0; j < m1Cols; j++) {
                matrix2D.doubles[i][j] = this.doubles[i][j] + vector.doubles[i];
            }
        }
        return matrix2D;
    }

    /**
     * Clones the matrix and then adds the provided value from every element in the matrix
     * @param val   the value
     * @return      the return matrix
     */
    public Matrix2D add(double val) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int rows = 0; rows < m1Rows; rows++) {
            for (int cols = 0; cols < m1Cols; cols++) {
                matrix2D.doubles[rows][cols] = this.doubles[rows][cols] + val;
            }
        }
        return matrix2D;
    }

    /**
     * Clones this matrix and the subtracts the provided matrix to it
     * @param m the matrix to subtract to the cloned matrix
     * @return  the new matrix
     */
    public Matrix2D sub(Matrix2D m) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        int m2Rows = m.getRowCount();
        int m2Cols = m.getColumnCount();
        if (m1Rows != m2Rows) {
            throw new IllegalArgumentException("m1.rows != m2.rows " + m1Rows + " != " + m2Rows);
        } else if (m1Cols != m2Cols) {
            throw new IllegalArgumentException("m1.cols != m2.cols " + m1Cols + " != " + m2Cols);
        }
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int i = 0; i < m1Rows; i++) {
            for (int j = 0; j < m1Cols; j++) {
                matrix2D.doubles[i][j] = this.doubles[i][j] - m.doubles[i][j];
            }
        }
        return matrix2D;
    }

    /**
     * Subtracts a vector to the matrix, the vector is treated as a column vector
     * @param vector    ...
     * @return          the resulting Matrix2D
     */
    public Matrix2D sub(Vector vector) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        if (m1Rows != vector.doubles.length) {
            throw new IllegalArgumentException("a.rows != b.length: (" + m1Rows + " != " + vector.doubles.length + ")");
        }
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int i = 0; i < m1Rows; i++) {
            for (int j = 0; j < m1Cols; j++) {
                matrix2D.doubles[i][j] = this.doubles[i][j] - vector.doubles[i];
            }
        }
        return matrix2D;
    }

    /**
     * Clones the matrix and then subtracts the provided value from every element in the matrix
     * @param val   the value
     * @return      the return matrix
     */
    public Matrix2D sub(double val) {
        int m1Rows = this.getRowCount();
        int m1Cols = this.getColumnCount();
        final Matrix2D matrix2D = new Matrix2D(m1Rows, m1Cols);
        for (int rows = 0; rows < m1Rows; rows++) {
            for (int cols = 0; cols < m1Cols; cols++) {
                matrix2D.doubles[rows][cols] = this.doubles[rows][cols] - val;
            }
        }
        return matrix2D;
    }

    public Dimension shape() {
        return new Dimension(this.getColumnCount(), this.getRowCount());
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
