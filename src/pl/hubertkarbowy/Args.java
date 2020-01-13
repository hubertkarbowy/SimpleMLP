package pl.hubertkarbowy;

import com.beust.jcommander.Parameter;

public class Args {

  @Parameter(names = "-truelabel", description = "Label for positive examples")
  String truelabel;

  @Parameter(names = "-trainset", description = "Path to trainset")
  String trainset;

  @Parameter(names = "-validset", description = "Path to validset")
  String validset;

  @Parameter(names = "-testset", description = "Path to testset")
  String testset;

  @Parameter(names = "-outdir", description = "Path where to save the results")
  String outdir;

  @Parameter(names = "-pretrained", description = "Path to a pretrained model")
  String pretrained;

  // Actions:
  @Parameter(names = "-split", description = "Do the train-valid-test split")
  boolean split = false;

  @Parameter(names = "-train", description = "Run training and save serialized model in outdir")
  boolean train = false;

  @Parameter(names = "-runeval", description = "Run evaluation from a pretrained model and print results (requires -pretrained)")
  boolean runeval = false;

  // Train parameters:
  @Parameter(names = "-max-delta", description = "Set max delta of network")
  double maxDelta = 0.03;

  @Parameter(names = "-perc-change", description = "Set maximum percent of weights changing in one iteration")
  double percChange = 0.05;

  @Parameter(names = "-max-patience", description = "Set max patience")
  int maxPatience = 30;

  @Parameter(names = "-max-iter", description = "Set maximum number of iterations")
  int maxIter = 5000;

  @Parameter(names = "-layers", description = "Number of hidden layers")
  int layers = 1;

  @Parameter(names = "-neurons", description = "Number of neurons in every hidden layer")
  int neurons = 10;

  @Parameter(names = "-trains", description = "Number of neurons in every hidden layer")
  int trains = 1000;
}
