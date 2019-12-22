package pl.hubertkarbowy;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.Math;
import static pl.hubertkarbowy.Utils.*;

public class MLPNetwork {
    private int inputSize;
    private int numClasses;
    private int[] neurons; // number of neurons in each layer
    private float[][][] weights;
    private float[][] inputs;
    private byte[] gold;

    private float randomFloat() {
        float f = (float)Math.random();
        if (Math.random() <= 0.5) return f;
        else return -f;
    }

    public MLPNetwork(int[] layersDefinition) {
        this.inputSize = layersDefinition[0] + 1; // input layer plus 1.0f in the zeroeth index as a dummy bias variable x0
        this.numClasses = layersDefinition[layersDefinition.length-1]; // output layer
        this.neurons = layersDefinition;
        this.weights = new float[neurons.length-1][][]; // L - 1 weight matrices

        for (int layerNum=0; layerNum<neurons.length-1; layerNum++) {
            int numNeuronsInPreviousLayer = neurons[layerNum];
            int numNeuronsInNextLayer = neurons[layerNum+1];
            float[][] initWeights = new float[numNeuronsInNextLayer][numNeuronsInPreviousLayer + 1]; // + 1 because we keep the bias and variable weights together
            for (int i=0; i<initWeights.length; i++) {
                for (int j=0; j<initWeights[i].length; j++) {
                    initWeights[i][j] = randomFloat();
                }
            }
            this.weights[layerNum] = initWeights;
        }
    }

    public void setInputs(String dirPath) throws IOException {
        setInputs(new File(dirPath));
    }

    public void setInputs(File dirPathAsFile) throws IOException { // todo: positive and negative examples
        File[] imgList = dirPathAsFile.listFiles();
        int numExamples = imgList.length;
        this.inputs = new float[numExamples][];
        this.gold = new byte[numExamples];
        int fCounter = 0;
        for (File f : imgList) {
            System.out.println("Name: " + f.getAbsolutePath());
            BufferedImage rescaled = loadAndRescale(f, (int)Math.sqrt(inputSize)); // assume square images
            this.inputs[fCounter] = imgToInputs(rescaled);
            fCounter++;
        }
    }

    public void setInputs(float[][] inputs) { this.inputs = inputs; }

    public void setWeights(float[][][] weights) { this.weights = weights; }

    public float[] forwardSingle(float[] singleInput) { // default problem: single label binary classification.
        return forwardSingle(singleInput, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID);
    }

    public float[] forwardSingle(float[] singleInput, ActivationFunction hiddenActivation, ActivationFunction outputActivation) {
        float[] a = singleInput;
        for (int L=0; L<weights.length; L++) {
            float[][] W = weights[L];
            if (L == weights.length - 1) { // for the last layer
                a = matMul(W, a, 0, outputActivation);
            }
            else { // for all other layers
                a = matMul(W, a, 1, hiddenActivation);
            }
//            int addOne = (L == weights.length-1 ? 0 : 1);
//            a = matMul(W, a, addOne, ActivationFunction.SIGMOID);
        }
        return a;
    }

    public float binaryCrossEntropyLoss() { // todo: implement
        return 0.0f;
    }

    public float[][] getInputs() { return this.inputs; }

    public float[][][] getWeights() { return this.weights; }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("************* SIMPLE MLP NETWORK SUMMARY ***************\n");
        sb.append("NumLayers=" + (neurons.length - 1) + " (w/o input)\n");
        sb.append("NumTrainingExamples=" + (inputs.length) + "\n");
        sb.append("InputSize=" + (neurons[0]) + "\n");
        sb.append("Layers:\n");
        for (int i=0; i<neurons.length; i++) {
            sb.append("Layer" + i + ": " + neurons[i] + " neuron(s)\n");
        }
        return sb.toString();
    }


}
