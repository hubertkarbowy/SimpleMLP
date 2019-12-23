package pl.hubertkarbowy;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

import java.io.File;
import java.io.IOException;

import static pl.hubertkarbowy.Utils.*;

public class MLPNetworkTests {

    private MLPNetwork net = null;
    private final int pictureSize = 28;
    private final float[][] W = new float[][] { {0.01f, 0.02f, 0.03f, 0.04f, 0.05f}, {0.05f, 0.04f, 0.03f, 0.02f, 0.01f} }; // first weight is bias
    private final float[] x = new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}; // first input is dummy 1.0 value to use with bias weight

    @Before
    public void setUp() throws IOException {
        int[] layerDefs = new int[]{pictureSize*pictureSize, 15, 7, 3, 1};
        this.net = new MLPNetwork(layerDefs);
        File f = new File(MLPNetwork.class.getResource("train1").getFile());
        net.setInputs(f);
    }


    @Test
    public void layersTest() {
        float[][] inputs = net.getInputs();
        float[][][] weights = net.getWeights();

        Assert.assertTrue("Expected only one weight matrix, but found " + weights[weights.length-1].length, weights[weights.length - 1].length == 1);
        Assert.assertTrue("Expected 15 rows in weight matrix for the second layer, but found " +  weights[0].length, weights[0].length == 15);
        Assert.assertTrue("Expected weight matrix for the second layer to contain " + (pictureSize*pictureSize + 1) + " columns " +
                          "but found " + weights[0][0].length, weights[0][0].length == pictureSize*pictureSize + 1);
        Assert.assertTrue("Expected weight matrix for the fourth layer to contain 8 columns " +
                          "but found " + weights[3][0].length, weights[2][0].length == 8);
    }

    @Test
    public void matMulLinearTest() { // NO ACTIVATION FUNCTION
        float[] expectedRet = new float[] {0.55f, 0.35f};
        float[] actualRet = matMul(W, x, 0, ActivationFunction.LINEAR);
        Assert.assertArrayEquals(expectedRet, actualRet, 0.01f);

        expectedRet = new float[] {1.0f, 0.55f, 0.35f};
        actualRet = matMul(W, x, 1, ActivationFunction.LINEAR);
        Assert.assertArrayEquals(expectedRet, actualRet, 0.01f);
    }

    @Test
    public void matMulSigmoidTest() { // SIGMOID
        float[] expectedRet = new float[] {0.63f, 0.58f};
        float[] actualRet = matMul(W, x, 0, ActivationFunction.SIGMOID);
        Assert.assertArrayEquals(expectedRet, actualRet, 0.01f);

        expectedRet = new float[] {1.0f, 0.63f, 0.58f};
        actualRet = matMul(W, x, 1, ActivationFunction.SIGMOID);
        Assert.assertArrayEquals(expectedRet, actualRet, 0.01f);
    }

    @Test
    public void matMulSoftmaxText() {
        float[] expectedRet = new float[] {0.54f, 0.45f};
        float[] actualRet = matMul(W, x, 0, ActivationFunction.SOFTMAX);
        Assert.assertArrayEquals(expectedRet, actualRet, 0.01f);
    }

    @Test
    public void forwardTest() {
        float[][][] allW = new float[][][] { W,  // 2 x 5
                                             { {0.15f, 0.25f, 0.35f}, {0.35f, 0.25f, 0.15f} }, // 2 x 3
                                             { {0.15f, 0.25f, 0.35f} } // 1 x 3
                                           };
        net.setWeights(allW);
        float[] expectedRet = new float[] {0.6299f};
        float[] actualRet = net.forwardSingle(x);
        Assert.assertArrayEquals(expectedRet, actualRet, 0.001f);
    }

    @Test
    public void binaryCrossEntropyTest() {
        float[][] inputs = new float[][] {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {1.0f, 0.05f, 0.04f, 0.03f, 0.02f}};
        float[][][] allW = new float[][][] { W,  // 2 x 5
                { {0.15f, 0.25f, 0.35f}, {0.35f, 0.25f, 0.15f} }, // 2 x 3
                { {0.15f, 0.25f, 0.35f} } // 1 x 3
        };
        int[] gold = new int[]{0, 1};
        net.setInputs(inputs);
        net.setWeights(allW);
        net.setGold(gold);
        float expectedCost = 0.7294f;
        float actualCost = net.binaryCrossEntropyLoss();
        Assert.assertEquals(expectedCost, actualCost, 0.001f);
    }
}
