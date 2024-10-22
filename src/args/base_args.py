import argparse

class BaseArgs:
  def __init__(self):
    self.arg_parser = argparse.ArgumentParser()
    self.add_base_arguments()
    
  def add_base_arguments(self):
    self.arg_parser.add_argument(
      "--model_type",
      type=str,
      required=True,
      choices=["val_pred", "val_sub"],
      help="Type of model to train or test"
    )
    self.arg_parser.add_argument(
      "--output_dir",
      type=str,
      help="Directory to store the results"
    )
    self.arg_parser.add_argument(
      "--domain_name",
      type=str,
      help="Name of the domain",
      choices=["blocks", "blocks-on", "gripper", "delivery", "visitall"]
    )
    self.arg_parser.add_argument(
      "--batch_size",
      type=int,
      default=32,
      help="Batch size"
    )
    self.arg_parser.add_argument(
      "--debug",
      action="store_true",
      help="Run in debug mode, which means that Comet ML will not be used"
    )
    self.arg_parser.add_argument(
      "--num_workers",
      type=int,
      default=0,
      help="Number of workers for the data loader"
    )
