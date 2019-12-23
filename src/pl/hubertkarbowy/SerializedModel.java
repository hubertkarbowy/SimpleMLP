package pl.hubertkarbowy;

import java.io.Serializable;

public class SerializedModel implements Serializable {

    int[] neurons;
    float[][][] weights;
    float maxDelta;
    float percChange;
    int maxPatience;
    int maxIter;

    public SerializedModel() {
    }

    public SerializedModel(int[] neurons, float[][][] weights, float maxDelta, float percChange, int maxPatience, int maxIter) {
        this.neurons = neurons;
        this.weights = weights;
        this.maxDelta = maxDelta;
        this.percChange = percChange;
        this.maxPatience = maxPatience;
        this.maxIter = maxIter;
    }

}
