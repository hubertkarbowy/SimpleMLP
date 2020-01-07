package pl.hubertkarbowy;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class Utils {

    static void train_test_split(Path start, String trueClass, Path out,
                                 int numTrain, int numValid, int numTest) throws IOException { // assume f points to a directory containing subdirs with
        // positive example (trueClass) and negative examples (all others)
        List<Path> alles = Files.walk(start).collect(Collectors.toList());
        List<Path> pos = alles.stream().filter(p -> p.getParent().getFileName().toString().equals(trueClass)).collect(Collectors.toList());
        List<Path> neg = alles.stream().filter(p -> !p.getParent().getFileName().toString().equals(trueClass)).collect(Collectors.toList());
        System.out.println(neg);

        Collections.shuffle(pos);
        Collections.shuffle(neg);

        int last_idx = 0;
        String[] subdirs = {"train", "valid", "test"};
        for (String subdir : subdirs) {
            Files.createDirectories(out.resolve(subdir).resolve("pos"));
            Files.createDirectories(out.resolve(subdir).resolve("neg"));
            int shiftBy = subdir.equals("train") ? numTrain : subdir.equals("valid") ? numValid : numTest;
            List<Path> thisPos = pos.subList(last_idx, last_idx+shiftBy);
            List<Path> thisNeg = neg.subList(last_idx, last_idx+shiftBy);
            for (int i = 0; i < thisPos.size(); i++) {
                Files.copy(thisPos.get(i), out.resolve(subdir).resolve("pos").resolve("" + i + "_pic.jpg"));
                Files.copy(thisNeg.get(i), out.resolve(subdir).resolve("neg").resolve("" + i + "_pic.jpg"));
            }
            last_idx += shiftBy;
        }
    }

    static BufferedImage loadAndRescale(File f, int imgWidthAndHeight) throws IOException {
        BufferedImage img = ImageIO.read(f);
        BufferedImage rescaled = new BufferedImage(imgWidthAndHeight, imgWidthAndHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = rescaled.createGraphics();
        g2d.drawImage(img, 0, 0, imgWidthAndHeight, imgWidthAndHeight, null);
        g2d.dispose();
        return rescaled;
    }

    public static float[] imgToInputs(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getWidth();
        float[] pixels = new float[(w*h) + 1]; // plus 1 for bias
        pixels[0] = 1.0f; // x0 = 1.0f - dummy variable
        int pixelIdx = 1;
//        int pixelIdx = 0;
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                int pp = (img.getRGB(x,y) >> 16) & 0xff; // just the red channel
                pixels[pixelIdx] = pp / 255f;
                pixelIdx++;
            }
        }
        return pixels;
    }

    static int getGoldFromPathName(Path path, boolean isBinary) {
        if (isBinary) {
            if (path.getParent().endsWith("pos")) return 1;
            else if (path.getParent().endsWith("neg")) return 0;
            else throw new RuntimeException("Only `pos` or `neg` directory names are allowed.");
        }
        else {
            int classNum = Integer.parseInt(path.getParent().getFileName().toString());
            return classNum;
        }
    }

    // If addOne is set to 1, this function returns the product of W and x and adds a 1.0f in the zeroeth index
    // of the return array. This is done to avoid having separate array for bias weights.
    // This parameter should be set to 1 in all layers except the output.
    public static float[] matMul(float[][] W, float[] x, int addOne, ActivationFunction activation) {
        if (addOne != 0 && addOne != 1) throw new RuntimeException("Set `addOne` to 1 if a dummy x=1 neuron is to be added to output or 0 otherwise.");
        float[] ret = new float[W.length + addOne];
        if (addOne == 1) {
            ret[0] = 1.0f;
        }
        float expSum = 0.0f;
        for (int i=0; i<W.length; i++) {
            for (int j=0; j<W[i].length; j++) {
                ret[i+addOne] += (W[i][j]*x[j]);
            }
            expSum += Math.exp(ret[i+addOne]);
            switch (activation) {
                case SIGMOID:
                    ret[i+addOne] = 1.0f / (1 + (float)Math.exp(-ret[i+addOne]));
                    break;
                case TANH:
                    ret[i+addOne] = (float)Math.tanh(ret[i+addOne]);
                    break;
                case RELU:
                    ret[i+addOne] = ret[i+addOne] > 0 ? ret[i+addOne] : 0;
                    break;
            }
        }
        if (activation == ActivationFunction.SOFTMAX) {
            for (int i=addOne; i<ret.length; i++) {
                ret[i] = (float)Math.exp(ret[i]) / expSum;
            }
        }
        return ret;
    }
}
