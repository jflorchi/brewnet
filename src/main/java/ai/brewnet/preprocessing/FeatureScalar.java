package io.aviea.ml.preprocessing;

public class FeatureScalar {

    private double dataMax, dataMin, dataRange, min, scale;

    public FeatureScalar(double dataMax, double dataMin, double dataRange, double min, double scale) {
        this.dataMax = dataMax;
        this.dataMin = dataMin;
        this.dataRange = dataRange;
        this.min = min;
        this.scale = scale;
    }

    public double getDataMax() {
        return dataMax;
    }

    public void setDataMax(double dataMax) {
        this.dataMax = dataMax;
    }

    public double getDataMin() {
        return dataMin;
    }

    public void setDataMin(double dataMin) {
        this.dataMin = dataMin;
    }

    public double getDataRange() {
        return dataRange;
    }

    public void setDataRange(double dataRange) {
        this.dataRange = dataRange;
    }

    public double getMin() {
        return min;
    }

    public void setMin(double min) {
        this.min = min;
    }

    public double getScale() {
        return scale;
    }

    public void setScale(double scale) {
        this.scale = scale;
    }

    @Override
    public String toString() {
        return "FeatureScalar{" +
                "dataMax=" + dataMax +
                ", dataMin=" + dataMin +
                ", dataRange=" + dataRange +
                ", min=" + min +
                ", scale=" + scale +
                '}';
    }

}
