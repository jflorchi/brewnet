package io.aviea.ml.loading;

import io.aviea.ml.Sequential;
import io.aviea.ml.Tensor;
import io.aviea.ml.Tensor2D;
import io.aviea.ml.preprocessing.MinMaxScalar;
import org.json.JSONArray;
import org.json.JSONObject;
import org.ujmp.core.doublematrix.DenseDoubleMatrix2D;
import org.ujmp.core.doublematrix.DoubleMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TensorFlowLoader {

    private final JSONObject modelObject;

    private MinMaxScalar xScalar, yScalar;

    public TensorFlowLoader(JSONObject modelObject) {
        this.modelObject = modelObject;
    }

    /**
     * Keras Weight file format:
     * The weight format is a JSON array every 2 indexes is a layer
     * The 1st of the 2 indexes are the weights.
     * The weights are comprised of 1 JSON array for every input node, and each node contains the weights for all
     * outgoing synapses
     * The 2nd of the 2 indexes are the biases
     * This is just an array of doubles and the length of the bias array is the size of the next layer
     * <p>
     * Since as of now, this library will only support MLPs we only care out dense layers
     * If we come across a layer that isn't dense, dropout, or leakyrelu we will throw an error
     * I hope to extend this library one day to be a little more robust
     *
     * @return
     * @throws IOException
     */
    public Sequential buildModel() {
        try {
            final List<LayerConfig> layerConfigs = new ArrayList<>();
            final JSONObject configObject = this.modelObject.getJSONObject("model");
            final JSONArray layers = configObject.getJSONObject("config").getJSONArray("layers");
            for (int i = 0; i < layers.length(); i++) {
                final JSONObject layer = layers.getJSONObject(i);
                String layerType = layer.getString("class_name");
                if (layerType.equals("Dense")) {
                    final int unitCount = layer.getJSONObject("config").getInt("units");
                    final String activation = layer.getJSONObject("config").getString("activation");
                    layerConfigs.add(new LayerConfig(layerType, activation, unitCount));
                }
            }
            // add each of the weights and biases to layer configs then build
            final JSONArray weightContainer = this.modelObject.getJSONArray("weights");
            int count = 0;
            for (int i = 0; i < weightContainer.length(); i += 2) {
                JSONArray weights = weightContainer.getJSONArray(i);
                JSONArray biases = weightContainer.getJSONArray(i + 1);
                final int inputSize = weights.length();
                final int outputSize = biases.length();
                final double[][] weightDoubles = new double[outputSize][inputSize];
                for (int j = 0; j < inputSize; j++) {
                    final JSONArray object = weights.getJSONArray(j);
                    for (int k = 0; k < object.length(); k++) {
                        weightDoubles[k][j] = object.getDouble(k);
                    }
                }
                final double[] biasDoubles = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    biasDoubles[j] = biases.getDouble(j);
                }
//                final DenseDoubleMatrix2D weightMatrix = DoubleMatrix.Factory.importFromArray(weightDoubles);
//                final DenseDoubleMatrix2D biasMatrix = DoubleMatrix.Factory.importFromArray(biasDoubles);
                System.out.println(weightDoubles.length + " - " + weightDoubles[0].length);
                System.out.println(biasDoubles.length);
                final Tensor2D weightMatrix = new Tensor2D(weightDoubles);
                final Tensor2D biasMatrix = new Tensor2D(new double[][]{biasDoubles});
                final LayerConfig currentConfig = layerConfigs.get(count);
                currentConfig.setBiases(biasMatrix);
                currentConfig.setWeights(weightMatrix);
                count++;
            }
            return new ModelConfig(layerConfigs).buildModel();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public MinMaxScalar getXScalar() {
        if (this.xScalar != null) {
            return this.xScalar;
        }
        return this.xScalar = new MinMaxScalar(
                this.modelObject.getJSONObject("scalars").getJSONObject("x").getJSONArray("x")
        );
    }

    public MinMaxScalar getYScalar() {
        if (this.yScalar != null) {
            return this.yScalar;
        }
        return this.yScalar = new MinMaxScalar(
                this.modelObject.getJSONObject("scalars").getJSONObject("y").getJSONArray("y")
        );
    }

}
