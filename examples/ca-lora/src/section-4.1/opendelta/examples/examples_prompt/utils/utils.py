
import json
import os
import re



# class EvalPrediction(NamedTuple):
#     """
#     Evaluation output (always contains labels), to be used to compute metrics.
#     Parameters:
#         predictions (:obj:`np.ndarray`): Predictions of the model.
#         label_ids (:obj:`np.ndarray`): Targets to be matched.
#         data_info: (:obj:`Dict[str, Any]`): Extra dataset information, one requires
#         to performs the evaluation. The data_info is a dictionary with keys from
#         train, eval, test to specify the data_info for each split of the dataset.
#     """
#     predictions: Union[np.ndarray, Tuple[np.ndarray]]
#     label_ids: np.ndarray
#     data_info: Dict[str, Any]

def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_last_checkpoint(output_dir):
    if os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
        return output_dir
    return None





def save_json(filepath, dictionary):
   with open(filepath, "w") as outfile:
      json.dump(dictionary, outfile)


def read_json(filepath):
   f = open(filepath,)
   return json.load(f)