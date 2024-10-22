from args.base_args import BaseArgs

class TrainingArgs(BaseArgs):
  def __init__(self):
    super().__init__()
    self.add_arguments()
  
  def get_args(self):
    args = self.arg_parser.parse_args()
    return args
    
  def add_arguments(self):
    self.arg_parser.add_argument(
      "--model_name",
      type=str,
      required=True,
      help="Name of the model. Will be used when storing the model"
    )
    self.arg_parser.add_argument(
      "--data_folder",
      type=str,
      required=True,
      help="Path to the data folder containing the data. Needs to contain the files 'train.json' and 'dev.json'"
    )
    self.arg_parser.add_argument(
      "--model_path",
      type=str,
      default=None,
      help="Path to model to load, if training is supposed to start from checkpoint."
    )
    self.arg_parser.add_argument(
      "--dropout",
      type=float,
      default=0.2,
      help="Dropout rate"
    )
    self.arg_parser.add_argument(
      "--experiment_name",
      type=str,
      help="Name of the experiment, which will show up in Comet ML"
    )
    self.arg_parser.add_argument(
      "--use_readout",
      action="store_true",
      help="If True, the GNN network will use a readout function"
    )
    self.arg_parser.add_argument(
      "--patience",
      type=int,
      help="Patience for early stopping"
    )
    self.arg_parser.add_argument(
      "--clip_grad",
      action="store_true",
      help="If True, will clip the gradient norm"
    )
    self.arg_parser.add_argument(
      "--use_lrs",
      action="store_true",
      help="If True, will use learning rate scheduler"
    )
    self.arg_parser.add_argument(
      "--permute_colors",
      action="store_true",
      help="If True, will permute the colors in the state and goal atoms. It is used to that all color MLPs will be trained."
    )
    self.arg_parser.add_argument(
      "--init_only_zeros",
      action="store_true",
      help="If True, will initialize the node states with zeros only"
    )