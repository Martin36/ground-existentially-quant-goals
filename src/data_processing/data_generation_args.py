import argparse

from constants import COLOR_PREDICATES

class DataGenerationArgs:
  def __init__(self):
    self.arg_parser = argparse.ArgumentParser()
    self.add_arguments()

  def get_args(self):
    args = self.arg_parser.parse_args()
    self.check_args(args)
    return args
  
  def check_args(self, args):   
    if args.nr_colors and args.nr_colors > len(COLOR_PREDICATES):
      raise ValueError(f"Only supporting up to '{len(COLOR_PREDICATES)}' colors, but got '{args.nr_colors}'")
    
    if args.nr_vars_per_type and args.max_nr_vars:
      total_nr_vars = sum([int(nr) for nr in args.nr_vars_per_type.split(",")])
      if total_nr_vars > args.max_nr_vars:
        raise ValueError(f"Total number of variables ({total_nr_vars}) in nr_vars_per_type cannot be larger than max_nr_vars ({args.max_nr_vars})")
      
    if args.split_sizes:
      print("'--split_sizes' argument is provided. Overriding '--split_ratio' argument")
      split_sizes = [int(nr) for nr in args.split_sizes.split(",")]
      if args.dataset_size and sum(split_sizes) != args.dataset_size:
        raise ValueError(f"Sum of split sizes ({sum(split_sizes)}) must be equal to dataset size ({args.dataset_size})")
      
    if not args.split_sizes and not args.dataset_size:
      raise ValueError("Either 'split_sizes' or 'dataset_size' must be provided")
    
    if args.nr_obj_split:
      nr_obj_splits = [int(nr) for nr in args.nr_obj_split.split(",")]

      if len(nr_obj_splits) != 3:
        raise ValueError(f"Invalid nr_obj_split argument: {args.nr_obj_split}")
      if nr_obj_splits[-1] > args.max_nr_objects:
        raise ValueError(f"Invalid nr_obj_split argument: {args.nr_obj_split}. The last number should be less than or equal to max_nr_objects ({args.max_nr_objects})")
            
    if args.gen_test_data and args.split_sizes:
      raise ValueError("The argument 'gen_test_data' cannot be used with 'split_sizes'")
    
    if args.gen_test_data and args.nr_obj_split:
      raise ValueError("The argument 'gen_test_data' cannot be used with 'nr_obj_split'")
    
    if args.gen_test_data and not args.dataset_size:
      raise ValueError("The argument 'gen_test_data' requires the argument 'dataset_size'")
    
    if not args.output_folder:
      if not args.max_nr_vars:
        raise ValueError("To use default folder, the argument 'max_nr_vars' must be provided")
      if not args.max_nr_objects:
        raise ValueError("To use default folder, the argument 'max_nr_objects' must be provided")
      if not args.dataset_size:
        raise ValueError("To use default folder, the argument 'dataset_size' must be provided")
      
      
  def add_arguments(self):
    self.arg_parser.add_argument(
      "--domain", type=str, required=True, help="Domain name"
    )
    self.arg_parser.add_argument(
      "--dataset_size", type=int, help="Number of instances to be generated"
    )
    self.arg_parser.add_argument(
      "--input_folder", type=str, help="Path to the input folder"
    )
    self.arg_parser.add_argument(
      "--output_folder", type=str, help="Path to the output folder"
    )
    self.arg_parser.add_argument(
      "--max_nr_vars", type=int, help="Maximum number of variables in the goal"
    )
    self.arg_parser.add_argument(
      "--goal_pred", type=str, help="Predicate to add in the goal"
    )
    self.arg_parser.add_argument(
      "--nr_goals", type=int, required=True, help="Number of goal atoms in the quantified goal"
    )
    self.arg_parser.add_argument(
      "--split_ratio", type=str, default="0.8,0.1,0.1", help="Ratio of train/dev/test splits"
    )
    self.arg_parser.add_argument(
      "--split_sizes", type=str, help="Comma separated list of number of instances for each split of the train/dev/test splits"
    )
    self.arg_parser.add_argument(
      "--max_nr_objects", type=int, help="The maximum number of objects to include in the dataset e.g. the dataset will only contain states which has at max this number of constants"
    )
    self.arg_parser.add_argument(
      "--nr_obj_split", type=str, help="Comma-separated string of the max number of objects in each split e.g. 5,6,8 means that train split will have max 5 objects, dev split only 6 objects and test split 7-8 objects"
    )
    self.arg_parser.add_argument(
      "--var_types", type=str, help="Comma separated list of variable types to include in the quantified goal, e.g. 'package,cell' for the delivery domain"
    )
    self.arg_parser.add_argument(
      "--nr_vars_per_type", type=str, help="Comma separated list of the number of variables of each type to include in the quantified goal, e.g. '2,1' for 2 vars of the first type and 1 for the second. This argument also need the '--var_types' argument to know which types these numbers belong to."
    )
    self.arg_parser.add_argument(
      "--remove_too_small_problems", action="store_true", help="If true, problems with less objects than variables will be removed."
    )
    self.arg_parser.add_argument(
      "--color_types", type=str, help="Comma separated list of object types that should have colors"
    )
    self.arg_parser.add_argument(
      "--nr_colors", type=int, help="The number of colors to use. Required if '--add_colors' is True"
    )
    self.arg_parser.add_argument(
      "--max_expanded_states", type=int, default=1000000, help="The maximum number of states to expand. If a problem has more states than this, no states will be used from this problem. Default is 1000000"
    )
    self.arg_parser.add_argument(
      "--partially_grounded", action="store_true", help="If set, the goals will be partially grounded. Only applies for the supervised dataset"
    )
    self.arg_parser.add_argument(
      "--gen_test_data", action="store_true", help="If set, it will generate only test data, which is the same as the other data but without the costs for the groundings"
    )
    self.arg_parser.add_argument(
      "--recolor_states", action="store_true", help="If set, the states will be recolored after they are sampled"
    )
    self.arg_parser.add_argument(
      "--use_neq_preds", action="store_true", help="If set, then 'neq' predicates between the constants will be added"
    )
