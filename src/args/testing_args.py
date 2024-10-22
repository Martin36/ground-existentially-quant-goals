from args.base_args import BaseArgs


class TestingArgs(BaseArgs):
  def __init__(self):
    super().__init__()
    self.add_arguments()
  
  def get_args(self):
    args = self.arg_parser.parse_args()
    self.check_args(args)
    return args
  
  def check_args(self, args):
    if not args.model_path and not args.models_folder and not args.random_sub:
      raise ValueError("Either 'model_path' or 'models_folder' must be provided")
    if not args.data_file and not args.data_folder:
      raise ValueError("Either 'data_file' or 'data_folder' must be provided")
    if args.data_file and args.data_folder:
      raise ValueError("Only one of 'data_file' or 'data_folder' must be provided")    
    if args.random_sub and not args.model_type == "val_sub":
      raise ValueError("The argument 'random_sub' can only be used with 'val_sub' models")      
  
  def add_arguments(self):
    self.arg_parser.add_argument(
      "--data_folder",
      type=str, 
      help="Path to the data folder. Need to contain 'test.json' file"
    )
    self.arg_parser.add_argument(
      "--data_file",
      type=str,
      help="Path to the data file. If provided, will ignore 'data_folder'"
    )
    self.arg_parser.add_argument(
      "--model_path",
      type=str,
      help="Path to model to test"
    )
    self.arg_parser.add_argument(
      "--models_folder",
      type=str,
      help="Path to folder containing models to test. If provided, will test all models in the folder"
    )
    self.arg_parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed"
    )
    self.arg_parser.add_argument(
      "--random_sub",
      action="store_true",
      help="Whether to randomly substitute variables in the goal"
    )
    self.arg_parser.add_argument(
      "--nr_objects",
      type=int,
      help="If set, will only test on instances with 'nr_objects' objects"
    )
