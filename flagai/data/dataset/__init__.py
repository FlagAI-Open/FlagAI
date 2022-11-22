from flagai.data.dataset.data_collator.collate_fn import ConstructBlockStrategy

from .block.dataset import BlockDataset
from .data_collator.collate_fn import (ConstructSeq2seqStrategy,
                                       ConstructSuperglueStrategy)
from .language_model.dataset import LambadaDataset, LMDataset
from .seq2seq.dataset import BlankLMDataset, ExtractionDataset, Seq2SeqDataset
from .superglue.dataset import SuperGlueDataset
