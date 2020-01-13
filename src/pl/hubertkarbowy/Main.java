package pl.hubertkarbowy;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

import com.beust.jcommander.*;

import static pl.hubertkarbowy.Utils.imgToInputs;
import static pl.hubertkarbowy.Utils.loadAndRescale;
import static pl.hubertkarbowy.Utils.train_test_split;

public class Main {
    final static int imgWidthAndHeight = 28;
    final static int[] layerDefs = new int[]{imgWidthAndHeight*imgWidthAndHeight, 1};
    static MLPNetwork net = new MLPNetwork(layerDefs);

    public static void trainAndSave
            ( String trainDir
            , String outFile
            , double maxDelta
            , double maxPercent
            , int maxPatience
            , int maxIter
            , int layers
            , int neurons
    ) throws IOException {
        int layersDef[] = new int[layers + 2];
        for (int i = 0; i < layersDef.length; ++i) {
            layersDef[i] = neurons;
        }
        layersDef[0] = imgWidthAndHeight*imgWidthAndHeight;
        layersDef[layersDef.length - 1] = 1;

        net = new MLPNetwork(layersDef, maxDelta, maxPercent, maxPatience, maxIter);
        net.setInputs(trainDir);
        System.out.println(net.toString());
        net.train();
        System.out.println("Loss = " + net.binaryCrossEntropyLoss());
        net.saveModel(outFile);
    }

    public static void runPredictions(String modelPath, String testDir) throws ClassNotFoundException, IOException {
        net = net.restoreModel(modelPath);
        List<Path> allTestFiles = Files.walk(Paths.get(testDir))
                                       .filter(p -> (p.getParent().endsWith("pos") || p.getParent().endsWith("neg")))
                                       .collect(Collectors.toList());
        int tp=0, fp=0, fn=0;
        int actp=0, actn=0;
        float prec=0.0f, recall=0.0f, f1=0.0f;

        for (Path imgPath : allTestFiles) {
            BufferedImage img = loadAndRescale(imgPath.toFile(), imgWidthAndHeight);
            float inputs[] = imgToInputs(img);
            boolean prediction = net.predictBinary(inputs);
            System.out.println(imgPath + " => " + prediction);
            if (imgPath.getParent().endsWith("pos")) {
                actp++;
                if (prediction == true) tp++;
                else fn++;
            }
            else if (imgPath.getParent().endsWith("neg")) {
                actn++;
                if (prediction == true) fp++;
            }
        }
        prec = (float)tp / (tp + fp);
        recall = (float)tp / (tp + fn);
        f1 = 2*((prec*recall)/(prec + recall));
        System.out.println("SUMMARY RESULTS: ");
        System.out.println("   Precision: " + tp + "/" + (tp + fp) + " = " + prec);
        System.out.println("      Recall: " + tp + "/" + (tp + fn) + " = " + recall);
        System.out.println("          F1: " + f1);
    }


    /*
    Uzycie:
    0) Sciagnac MNIST as JPG stad: https://www.kaggle.com/scolianni/mnistasjpg i przeniesc gdzies
    1) Przygotowanie zbiorow treningowych, walidacyjnych (na razie nie uzywane) i testowych. Program arguments:

       -trainset /home/hubert/wymiana/Projekty/Studia/MY1wlasne/wrn/mnist_all/trainingSet/trainingSet/
       -truelabel "3"      <---------------------- tutaj cyfra do rozpoznania
       -outdir /home/hubert/wymiana/Projekty/Studia/MY1wlasne/wrn/wyj-train1000
       -split

     2) Trening (puscic kilka razy najlepiej a potem srednie wyniki liczyc przy ewaluacji). Program arguments:
        -trainset /home/hubert/wymiana/Projekty/Studia/MY1wlasne/wrn/wyj-train1000/train/
        -outdir /home/hubert/wymiana/Projekty/Studia/MY1wlasne/wrn/ser1000-1.ser
        -train

     3) Ewaluacja. Program arguments:
        -testset /home/hubert/wymiana/Projekty/Studia/MY1wlasne/wrn/wyj-train1000/test/
        -pretrained /home/hubert/wymiana/Projekty/Studia/MY1wlasne/wrn/ser1000-1.ser
        -runeval
     */

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        Args cmdargs = new Args();
        JCommander.newBuilder().addObject(cmdargs).build().parse(args);
        if (cmdargs.split) {
            train_test_split(Paths.get(cmdargs.trainset), cmdargs.truelabel, Paths.get(cmdargs.outdir), cmdargs.trains, 20, 50);
        }
        else if (cmdargs.train) {
            trainAndSave
                    ( cmdargs.trainset
                    , cmdargs.outdir
                    , cmdargs.maxDelta
                    , cmdargs.percChange
                    , cmdargs.maxPatience
                    , cmdargs.maxIter
                    , cmdargs.layers
                    , cmdargs.neurons);
        }
        else if (cmdargs.runeval) {
            runPredictions(cmdargs.pretrained, cmdargs.testset);
        }
    }
}
