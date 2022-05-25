from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSeq2Seq
from flagai.data.tokenizer import GLMLargeEnWordPieceTokenizer, GLMLargeChTokenizer
from flagai.data.dataset import Seq2SeqDataset
from flagai.test_utils import Seq2SeqCollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, CH_TASKS
from flagai.data.dataset import ConstructSeq2seqStrategy
import unittest


class TrainerTestCase(unittest.TestCase):

    def test_init_trainer_pytorch(self):
        # Compared with original seq2seq, seq2seq dataset is used
        # task_name :['cmrc',xxxx]
        task_name = "cmrc"

        cl_args = Seq2SeqCollateArguments()
        trainer = Trainer(env_type='pytorch',
                          epochs=1,
                          batch_size=4,
                          eval_interval=5,
                          log_interval=50,
                          experiment_name='glm_large',
                          pytorch_device='cpu',
                          load_dir=None,
                          lr=1e-4)
        print("downloading...")

        if task_name in CH_TASKS:
            tokenizer = GLMLargeChTokenizer(add_block_symbols=True,
                                            add_task_mask=True,
                                            add_decoder_mask=False,
                                            fix_command_token=True)
            model_name = 'GLM-large-ch'
        else:
            tokenizer = GLMLargeEnWordPieceTokenizer(
                tokenizer_model_type='bert-base-chinese')
            model_name = 'GLM-large-en'

        train_dataset = Seq2SeqDataset(task_name=task_name,
                                       data_dir='./datasets/',
                                       dataset_type='train',
                                       tokenizer=tokenizer)
        valid_dataset = Seq2SeqDataset(task_name=task_name,
                                       data_dir='./datasets/',
                                       dataset_type='dev',
                                       tokenizer=tokenizer)
        collate_fn = ConstructSeq2seqStrategy(cl_args,
                                                tokenizer,
                                                task_name=task_name)
        train_dataset.example_list = train_dataset.example_list[:20]
        valid_dataset.example_list = valid_dataset.example_list[:20]

        model = GLMForSeq2Seq.from_pretrain(model_name=model_name)

        trainer.train(model,
                      collate_fn=collate_fn,
                      train_dataset=train_dataset,
                      valid_dataset=valid_dataset,
                      metric_methods=DEFAULT_METRICS[task_name])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TrainerTestCase('test_init_trainer_pytorch'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
