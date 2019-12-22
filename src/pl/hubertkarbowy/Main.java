package pl.hubertkarbowy;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
	    int[] layerDefs = new int[]{28*28, 1};
        MLPNetwork net = new MLPNetwork(layerDefs);
        net.setInputs("/home/hubert/allrepos/SimpleMLP/other/train1");
        // net.forwardSingle(net.getInputs()[0]);
        System.out.println(net.toString());
    }
}
