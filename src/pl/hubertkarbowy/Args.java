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
}
