package io.aviea.ml.preprocessing;

import io.aviea.ml.Tensor2D;
import org.json.JSONArray;
import org.json.JSONObject;
import org.ujmp.core.doublematrix.DenseDoubleMatrix2D;

import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("Duplicates")
public class MinMaxScalar {

    private final List<FeatureScalar> featureScalars = new ArrayList<>();

    public MinMaxScalar(JSONArray scalar) {
        try {
            for (int i = 0; i < scalar.length(); i++) {
                final JSONObject featureScalar = scalar.getJSONObject(i);
                final double dataMax = featureScalar.getDouble("data_max");
                final double dataMin = featureScalar.getDouble("data_min");
                final double dataRange = featureScalar.getDouble("data_range");
                final double min = featureScalar.getDouble("min");
                final double scale = featureScalar.getDouble("scale");
                this.featureScalars.add(new FeatureScalar(dataMax, dataMin, dataRange, min, scale));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void mutateTransform(final Tensor2D matrix) {
        if (matrix.getColumnCount() != this.featureScalars.size()) {
            System.out.println("Matrix does not fit the size of the feature scalars");
            return;
        }
        for (int i = 0; i < matrix.getRowCount(); i++) {
            for (int j = 0; j < matrix.getColumnCount(); j++) {
                final FeatureScalar scalar = this.featureScalars.get(j);
                double item = matrix.tensors[i].doubles[j];
                item *= scalar.getScale();
                item += scalar.getMin();
                matrix.tensors[i].doubles[j] = item;
            }
        }
    }

    public void mutateInverseTransform(final Tensor2D matrix) {
        if (matrix.getRowCount() != this.featureScalars.size()) {
            System.out.println("Matrix does not fit the size of the feature scalars");
            return;
        }
        for (int i = 0; i < matrix.getRowCount(); i++) {
            final FeatureScalar scalar = this.featureScalars.get(i);
            for (int j = 0; j < matrix.getColumnCount(); j++) {
                double item = matrix.tensors[i].doubles[j];
                item -= scalar.getMin();
                item /= scalar.getScale();
                matrix.tensors[i].doubles[j] = item;
            }
        }
    }

}
