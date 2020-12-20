package ai.brewnet;

import java.util.Arrays;
import java.util.Random;

public class Vector {

    public final double[] doubles;

    public Vector(double[] doubles) {
        this.doubles = doubles;
    }

    public Vector(int size) {
        this.doubles = new double[size];
    }

    /**
     * Creates a Vector of zeros
     * @param length    size of vector
     * @return          the vector
     */
    public static Vector createZeros(int length) {
        final double[] dubs = new double[length];
        for (int i = 0; i < length; i++) {
            dubs[i] = 0;
        }
        return new Vector(dubs);
    }


    /**
     * Creates a Vector of random double values
     * @param length    size of vector
     * @return          the vector
     */
    public static Vector createRandom(int length) {
        final double[] dubs = new double[length];
        for (int i = 0; i < length; i++) {
            dubs[i] = new Random().nextDouble();
        }
        return new Vector(dubs);
    }

    /**
     * Multiplies every element of the vector by the scalar
     * @param d scalar
     * @return  scaled vector
     */
    public Vector scale(final double d) {
        final Vector v = new Vector(this.doubles.length);
        for (int i = 0; i < this.doubles.length; i++) {
            v.doubles[i] = this.doubles[i] * d;
        }
        return v;
    }

    /**
     *
     * @param vector ...
     * @return  a new vector that is the result of the addition
     */
    public Vector add(final Vector vector) {
        if (this.doubles.length != vector.doubles.length) {
            throw new IllegalArgumentException("Vector length miss match " + this.doubles.length + " != " + vector.doubles.length);
        }
        final Vector v = new Vector(this.doubles.length);
        for (int i = 0; i < this.doubles.length; i++) {
            v.doubles[i] = this.doubles[i] + vector.doubles[i];
        }
        return v;
    }

    /**
     *
     * @param vector ...
     * @return  a new vector that is the result of the subtraction
     */
    public Vector sub(final Vector vector) {
        if (this.doubles.length != vector.doubles.length) {
            throw new IllegalArgumentException("Vector length miss match " + this.doubles.length + " != " + vector.doubles.length);
        }
        final Vector v = new Vector(this.doubles.length);
        for (int i = 0; i < this.doubles.length; i++) {
            v.doubles[i] = this.doubles[i] - vector.doubles[i];
        }
        return v;
    }

    /**
     * @param vector    parameter
     * @return  a double, matrix multiplication with inferred bounds
     */
    public double mul(final Vector vector) {
        if (this.doubles.length != vector.doubles.length) {
            throw new IllegalArgumentException("Vector length miss match " + this.doubles.length + " != " + vector.doubles.length);
        }
        double val = 0;
        for (int i = 0; i < this.doubles.length; i++) {
            val += this.doubles[i] * vector.doubles[i];
        }
        return val;
    }

    /**
     * Converts the vector a Matrix where this.doubles.length == Matrix.getRowCount() and Matrix.getColumnCount() == 1
     * @return  the matrix
     */
    public Matrix2D toMatrix() {
        final Matrix2D matrix2D = new Matrix2D(this.doubles.length, 1);
        for (int i = 0; i < this.doubles.length; i++) {
            matrix2D.doubles[i][0] = this.doubles[i];
        }
        return matrix2D;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Vector vector = (Vector) o;
        return Arrays.equals(doubles, vector.doubles);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(doubles);
    }

    @Override
    public String toString() {
        return Arrays.toString(this.doubles);
    }
}
