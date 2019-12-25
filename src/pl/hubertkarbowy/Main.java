package pl.hubertkarbowy;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static pl.hubertkarbowy.Utils.imgToInputs;
import static pl.hubertkarbowy.Utils.loadAndRescale;

public class Main {
    final static int imgWidthAndHeight = 28;
    final static int[] layerDefs = new int[]{imgWidthAndHeight*imgWidthAndHeight, 1};
    static MLPNetwork net = new MLPNetwork(layerDefs);
    final static String currentDir = System.getProperty("user.dir") + "/";

    public static void trainAndSave() throws IOException {
        net.setInputs(currentDir + "other/train2");
        System.out.println(net.toString());
        net.train();
        System.out.println("Loss = " + net.binaryCrossEntropyLoss());
        net.saveModel(currentDir + "other/pretrained2.ser");
    }

    public static void runPredictions() throws IOException, ClassNotFoundException {
        String[] images = {currentDir + "other/test1/three_11.png",
                           currentDir + "other/test1/one_2.png",
                           currentDir + "other/test1/eight_2.png"};
        net.restoreModel(currentDir + "other/pretrained2.ser");
        for (String imgPath : images) {
            BufferedImage img = loadAndRescale(new File(imgPath), imgWidthAndHeight);
            float inputs[] = imgToInputs(img);
            System.out.println(imgPath + " => " + net.predictBinary(inputs));
        }
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        trainAndSave();
        runPredictions();
    }
}
