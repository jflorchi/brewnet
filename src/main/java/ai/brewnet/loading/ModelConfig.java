package io.aviea.ml.loading;

import io.aviea.ml.Sequential;
import io.aviea.ml.layers.Layer;

import java.util.ArrayList;
import java.util.List;

public class ModelConfig {

    private final List<LayerConfig> layerConfigs;

    public ModelConfig(final List<LayerConfig> layerConfigs) {
        this.layerConfigs = layerConfigs;
    }

    /**
     * Builds a model based on the configuration
     *
     * @return the model
     */
    public Sequential buildModel() {
        // Construct the layers
        final List<Layer> layers = new ArrayList<>();
        this.layerConfigs.forEach(config -> layers.add(config.buildLayer()));
        // Link the layers
        for (int i = 0; i < layers.size(); i++) {
            if (i == layers.size() - 1) {
                final Layer cur = layers.get(i);
                final Layer prev = layers.get(i - 1);
                cur.setInputLayer(prev);
                prev.setOutputLayer(cur);
            } else {
                final Layer cur = layers.get(i);
                final Layer next = layers.get(i + 1);
                cur.setOutputLayer(next);
                next.setInputLayer(cur);
            }
        }
        // Pass them into a new MLP and return
        return new Sequential(layers);
    }
}
