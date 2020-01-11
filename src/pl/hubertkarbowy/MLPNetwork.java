package pl.hubertkarbowy;

import java.awt.image.BufferedImage;
import java.io.*;
import java.lang.Math;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static pl.hubertkarbowy.Utils.*;

public class MLPNetwork {
    private int inputSize;
    private int[] neurons; // number of neurons in each layer
    private float[][][] weights;
    private float[][] inputs;
    private int[] gold;
    private boolean isBinary = true;  // for binary classification we use `pos` / `neg` dir names,
                                      // for multi-class we use `0`, `1`, `2` etc.
    private boolean isTest = false;
    private float maxDelta = 0.03f;   // by how much at most will we increase/decrease each weight
    private float percChange = 0.05f; // what percentage of weights to randomly change on each RHC iteration
    private int maxPatience = 30;     // number of RHC iterations after which to stop if no improvement was made
    private int maxIter = 5000;       // total maximum number of RHC iterations

    private float randomFloat() {
        float f = (float)Math.random();
        if (Math.random() <= 0.5) return f;
        else return -f;
    }

    public MLPNetwork(int[] layersDefinition) {
        this.inputSize = layersDefinition[0] + 1; // input layer plus 1.0f in the zeroeth index as a dummy bias variable x0
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

    public MLPNetwork(int[] layersDefinition, double maxDelta, double percChange, int maxPatient, int maxIter) {
        this(layersDefinition);
        this.maxDelta = (float)maxDelta;
        this.percChange = (float)percChange;
        this.maxPatience = maxPatient;
        this.maxIter = maxIter;
    }

    public void setInputs(String dirPath) throws IOException {
        setInputs(new File(dirPath));
    }

    public void setInputs(File dirPathAsFile) throws IOException {
        File[] trainsetPath = dirPathAsFile.listFiles();
        List<String> dirNames = Arrays.asList(trainsetPath).stream().map(x -> x.getName()).collect(Collectors.toList());
        try {
            if (dirNames.contains("pos") && dirNames.contains("neg")) {
                isBinary = true;
            }
            else if (dirNames.stream().map(elem -> Integer.parseInt(elem)).collect(Collectors.toList()).size() > 0) {
                isBinary = false;
            }
            else {
                throw new RuntimeException();
            }
        }
        catch (Exception e) {
            throw new RuntimeException("Please put your examples either in `pos` and `neg` directories, or in directories " +
                                       " whose names are integers representing the class number.");
        }
        List<Path> imgList = Files.walk(dirPathAsFile.toPath()).filter(Files::isRegularFile).collect(Collectors.toList());
        Collections.shuffle(imgList);
        int numExamples = imgList.size();
        this.inputs = new float[numExamples][];
        this.gold = new int[numExamples];
        int fCounter = 0;
        for (Path f : imgList) {
            System.out.println("Name: " + f.toString());
            BufferedImage rescaled = loadAndRescale(f.toFile(), (int)Math.sqrt(inputSize)); // assume square images
            this.inputs[fCounter] = imgToInputs(rescaled);
            this.gold[fCounter] = (isTest ? -1000 : getGoldFromPathName(f, isBinary));
            fCounter++;
        }
    }

    public void setInputs(float[][] inputs) { this.inputs = inputs; }

    public void setWeights(float[][][] weights) { this.weights = weights; }

    public void setGold(int[] gold) { this.gold = gold; }

    protected float[] forwardSingle(float[] singleInput) { // default problem: single label binary classification.
        return forwardSingle(singleInput, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID);
    }

    protected float[] forwardSingle(float[] singleInput, ActivationFunction hiddenActivation, ActivationFunction outputActivation) {
        float[] a = singleInput;
        for (int L=0; L<weights.length; L++) {
            float[][] W = weights[L];
            if (L == weights.length - 1) { // for the last layer
                a = matMul(W, a, 0, outputActivation);
            }
            else { // for all other layers
                a = matMul(W, a, 1, hiddenActivation);
            }
        }
        return a;
    }

    protected float binaryCrossEntropyLoss() {
        float crossEntropy = 0.0f;
        for (int n=0; n<inputs.length; n++) {
            float[] outputs = forwardSingle(inputs[n], ActivationFunction.SIGMOID, ActivationFunction.SIGMOID);
            float y = gold[n];
            float y_pred = outputs[0];
            crossEntropy += (y*Math.log(y_pred)) + ((1 - y)*Math.log(1 - y_pred));
        }
        return -crossEntropy / inputs.length;
    }

    // Trains the model and updates weights array in this class
    public void train() {
        Map<Integer, int[]> floatAddresses = new HashMap<>();

        int weightCounter = 0;
        for (int i=0; i<weights.length; i++) {
            for (int j=0; j<weights[i].length; j++) {
                for (int k=0; k<weights[i][j].length; k++) {
                    floatAddresses.put(weightCounter, new int[]{i, j, k});
                    weightCounter++;
                }
            }
        }
        System.out.println("This model has " + weightCounter + " parameters.");

        Random r = new Random();
        float bestLoss = binaryCrossEntropyLoss();
        int patience = 0;
        System.out.println("Starting with loss of " + bestLoss);
        for (int cnt=0; ; cnt++) {
            int[] randomWeightIndices = r.ints((int) (percChange * weightCounter), 0, weightCounter).toArray();
            float[] originalWeights = getWeightValues(floatAddresses, randomWeightIndices);
            float[] modifiedWeights = modifyFloats(originalWeights, r);
            changeWeights(floatAddresses, randomWeightIndices, modifiedWeights);
            float modifLoss = binaryCrossEntropyLoss();
            if (modifLoss < bestLoss) {
                System.out.println("On iteration " + cnt + " loss fell to " + modifLoss);
                bestLoss = modifLoss;
                patience = 0;
            }
            else {
                changeWeights(floatAddresses, randomWeightIndices, originalWeights); // restore original weights if the loss goes up
                patience++;
            }
            if (patience > maxPatience || cnt > maxIter) break;
        }
    }

    public boolean predictBinary(float[] single) {
        float res[] = forwardSingle(single, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID);
        if (res[0] >= 0.5) return true;
        else return false;
    }

    private float[] modifyFloats(float[] originalWeights, Random r) {
        float[] modifiedWeights = new float[originalWeights.length];
        for (int i=0; i<originalWeights.length; i++) {
            int factor = r.nextInt(5);
            if (factor == 0) factor = 1;
            float delta;
            if (Math.random() >= 0.5) {
                 delta = (-maxDelta / factor) * originalWeights[i];
            }
            else {
                delta = (maxDelta / factor) * originalWeights[i];
            }
            modifiedWeights[i] = originalWeights[i] + delta;
        }
        return modifiedWeights;
    }

    private float[] getWeightValues(Map<Integer, int[]> floatAddresses, int[] randomWeightIndices) {
        float[] currWeights = new float[randomWeightIndices.length];
        for (int n=0; n<randomWeightIndices.length; n++) {
            int[] addresses = floatAddresses.get(randomWeightIndices[n]);
            int i = addresses[0]; int j = addresses[1]; int k = addresses[2];
            currWeights[n] = this.weights[i][j][k];
        }
        return currWeights;
    }

    // Saves newValues to the weights array in this class
    private void changeWeights(Map<Integer, int[]> floatAddresses, int[] randomWeightIndices, float[] newValues) {
        for (int n=0; n<randomWeightIndices.length; n++) {
            int[] addresses = floatAddresses.get(randomWeightIndices[n]);
            int i = addresses[0]; int j = addresses[1]; int k = addresses[2];
            this.weights[i][j][k] = newValues[n];
        }
    }

    public void saveModel(String destPath) throws IOException {
        SerializedModel model = new SerializedModel(this.neurons, this.weights, this.maxDelta, this.percChange, this.maxPatience, this.maxIter);
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(destPath))) {
            oos.writeObject(model);
        }
    }

    public static MLPNetwork restoreModel(String srcPath) throws IOException, ClassNotFoundException {
        SerializedModel model = null;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(srcPath))) {
            model = (SerializedModel) ois.readObject();
        }
        int[] savedNeurons = model.neurons;
        MLPNetwork newNet = new MLPNetwork(savedNeurons);
        newNet.neurons = model.neurons;
        newNet.weights = model.weights;
        newNet.maxDelta = model.maxDelta;
        newNet.percChange = model.percChange;
        newNet.maxPatience = model.maxPatience;
        newNet.maxIter = model.maxIter;
        return newNet;
    }

    public int[] getNeurons() { return neurons; }

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
