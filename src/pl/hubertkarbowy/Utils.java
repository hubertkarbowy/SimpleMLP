package pl.hubertkarbowy;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Utils {
    static BufferedImage loadAndRescale(File f, int imgWidthAndHeight) throws IOException {
        BufferedImage img = ImageIO.read(f);
        BufferedImage rescaled = new BufferedImage(imgWidthAndHeight, imgWidthAndHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = rescaled.createGraphics();
        g2d.drawImage(img, 0, 0, imgWidthAndHeight, imgWidthAndHeight, null);
        g2d.dispose();
        return rescaled;
    }

    static float[] imgToInputs(BufferedImage img) {
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
