from .superglue.dataset import SuperGlueDataset
from .language_model.dataset import LambadaDataset, LMDataset
from .seq2seq.dataset import Seq2SeqDataset, BlankLMDataset, ExtractionDataset
from flagai.data.dataset.data_collator.collate_fn import ConstructBlockStrategy
from .block.dataset import BlockDataset
from .data_collator.collate_fn import ConstructSuperglueStrategy, ConstructSeq2seqStrategy
