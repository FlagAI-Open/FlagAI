# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.data.dataset.superglue.processor import *
from flagai.data.dataset.superglue.pvp import *

from flagai.metrics import qa_exact_match, qa_f1, accuracy_metric, f1_macro_metric, f1_metric, multirc_em

PROCESSOR_DICT = {
    "ax-b": AxBProcessor,
    "cb": CbProcessor,
    "copa": CopaProcessor,
    "multirc": MultiRcProcessor,
    "rte": RteProcessor,
    "wic": WicProcessor,
    "wsc": WscProcessor,
    "boolq": BoolQProcessor,
    "record": RecordProcessor,
    "ax-g": AxGProcessor,
    "afqmc": AFQMCProcessor,
    "tnews": TNewsProcessor,
    'wanke': WankeProcessor,
    "cmrc": CMRCProcessor,
    "cola": ColaProcessor,
    "sst2": Sst2Processor,
    "mrpc": MrpcProcessor,
    "qqp": QqpProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "qnli": QnliProcessor,
    "xstance": XStanceProcessor,
    "xstance-de": lambda: XStanceProcessor("de"),
    "xstance-fr": lambda: XStanceProcessor("fr"),
    "race": RaceProcessor,
    "agnews": AgnewsProcessor,
    "yahoo": YahooAnswersProcessor,
    "yelp-polarity": YelpPolarityProcessor,
    "yelp-full": YelpFullProcessor,
    "squad": SquadProcessor,
    'cluewsc': CLUEWSCProcessor,
}

PVPS = {
    'ax-b': RtePVP,
    'cb': CbPVP,
    'copa': CopaPVP,
    'multirc': MultiRcPVP,
    'rte': RtePVP,
    'wic': WicPVP,
    'wsc': WscPVP,
    'boolq': BoolQPVP,
    'record': RecordPVP,
    'ax-g': RtePVP,
    "afqmc": AFQMCPVP,
    'tnews': TNewsPVP,
    'cluewsc': CLUEWSCPVP,
    'wanke': WankePVP,
    'cmrc': CMRCPVP,
    'sst2': Sst2PVP,
    'cola': ColaPVP,
    'mrpc': MrpcPVP,
    'qqp': QqpPVP,
    'mnli': MnliPVP,
    'qnli': QnliPVP,
    'squad': SquadPVP,
    'race': RacePVP,
    'agnews': AgnewsPVP,
    'yelp-polarity': YelpPolarityPVP,
    'yelp-full': YelpFullPVP,
    'yahoo': YahooPVP,
    'xstance': XStancePVP,
    'xstance-de': XStancePVP,
    'xstance-fr': XStancePVP,
}

DEFAULT_METRICS = {
    "record": [("EM", qa_exact_match), ("F1", qa_f1)],
    "copa": [("accuracy", accuracy_metric)],
    "rte": [("accuracy", accuracy_metric)],
    "boolq": [("accuracy", accuracy_metric)],
    "wic": [("accuracy", accuracy_metric)],
    "wsc": [("accuracy", accuracy_metric)],
    "cb": [("accuracy", accuracy_metric), ("f1-macro", f1_macro_metric)],
    "multirc": [("f1a", f1_metric), ("em", multirc_em),
                ("acc", accuracy_metric)],
    "mnli": [("accuracy", accuracy_metric)],
    "sst2": [("accuracy", accuracy_metric)],
    "qnli": [("accuracy", accuracy_metric)],
    "qqp": [("accuracy", accuracy_metric)],
    "mrpc": [("accuracy", accuracy_metric)],
    "cola": [("accuracy", accuracy_metric)],
    "squad": [("accuracy", accuracy_metric)],
    "afqmc": [("accuracy", accuracy_metric)],
    "tnews": [("accuracy", accuracy_metric)],
    "cluewsc": [("accuracy", accuracy_metric)],
    # "cmrc": [("f1a", f1_metric), ("em", multirc_em)],
    "cmrc": [],
    # "cmrc": [("accuracy", accuracy_metric)],
    "wanke": [("accuracy", accuracy_metric)],
    "pretrain": [],
    "title_generation": [],
}

urls = {
    'axb': 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-b.zip',
    'cb': 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip',
    'copa': 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip',
    'multirc':
    'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip',
    'rte': 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip',
    'wic': 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip',
    'wsc': 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip',
    'boolq': 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip',
    'record':
    'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip',
    'ax-g': 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-g.zip',
    "afqmc":
    "https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip",
    "tnews":
    "https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip",
    "cola": "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
    "cmrc":
    "https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip",
    "sst2": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    "mrpc": "https://www.microsoft.com/en-us/download/details.aspx?id=52398",
    "qqp": "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
    "mnli": "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
    "mnli-mm": "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
    "qnli": "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
}

MULTI_TOKEN_TASKS = ['copa', 'record', 'cmrc', 'wsc']

CH_TASKS = ['afqmc', 'tnews', 'cmrc', 'wanke', 'lang8_hsk']


class SuperGlueProcessor:

    def __init__(self):
        self.processdict = PROCESSOR_DICT

    def _check_files(self, dirname, dname):
        return os.path.exists(os.path.join(dirname, dname))

    def _download_data(self, dirname, dname):
        try:
            import requests
            print("downloading {} with requests".format(dname))
            url = urls[dname]
            print('url', url)
            r = requests.get(url)
            print('download successed!')

            zip_file = os.path.join(dirname, "tmp_" + dname + ".zip")
            if dname in ["afqmc", "tnews", "cmrc"]:
                dirname += "/" + dname

            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(zip_file, "wb") as code:
                code.write(r.content)
        except Exception:
            raise ConnectionError('Dataset downloading failure!')

        try:
            self._unzip_file(zip_file, dirname)
            os.remove(zip_file)
        except Exception:
            raise ValueError('file unzip failure!')
        files = [f for f in os.listdir(dirname)]

        for f in files:
            try:
                if f.lower() == dname:
                    os.rename(dirname + '/' + f, dirname + '/' + dname)
            except:
                pass

    def _unzip_file(self, src_file, dst_dir):
        r = zipfile.is_zipfile((src_file))
        if r:
            fz = zipfile.ZipFile(src_file, 'r')
            for file in fz.namelist():
                fz.extract(file, dst_dir)
        else:
            print("This is not zip")

    def get_processor(self, dirname, dname):
        if dname in self.processdict:
            # dirname is none means that we are processing collate function and datadir is not required
            if dirname is None or self._check_files(dirname, dname):
                return self.processdict[dname]
            else:
                self._download_data(dirname, dname)
                return self.processdict[dname]
        else:
            raise ValueError('Dataset not supported!')


class ExampleProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        # Assign the filename of train set
        return self._create_examples(os.path.join(data_dir, "train.tsv"),
                                     "train")

    def get_dev_examples(self, data_dir, for_train=False):
        # Assign the filename of dev set
        return self._create_examples(os.path.join(data_dir, "dev.tsv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        # Assign the filename of test set
        return self._create_examples(os.path.join(data_dir, "test.tsv"),
                                     "test")

    def get_labels(self):
        # Return all label categories
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        """
        Construct the Input example, which contains the following keys
        text_a (str, required): The content text
        text_b (str, optional): Usually the
        label (str, required): the labels
        guid (str, required): A unique id to one InputExample element
        """
        examples = []
        df = read_tsv(path)

        for idx, row in df.iterrows():
            guid = f"{set_type}-{idx}"
            text_a = punctuation_standardization(row['sentence'])
            label = row.get('label', None)
            example = InputExample(guid=guid, text_a=text_a, label=label)
            examples.append(example)
        return examples


class ExamplePVP(PVP):
    # Map the actual token (in original file) to the actual meaning of it
    VERBALIZER = {"0": ["中立"], "1": ["利好"], "2": ["利空"]}

    @staticmethod
    def available_patterns():
        # Return ids of all available patterns
        return [0]

    @property
    def is_multi_token(self):
        # If the label can contain more than 1 token, return True
        return True

    def get_parts(self, example: InputExample) -> FilledPattern:
        # Organize the elements in InputExample into a designed pattern
        text_a = self.shortenable(example.text_a)
        if self.pattern_id == 0:
            return ["标题：", text_a, "类别：", [self.mask]], []
        else:
            raise NotImplementedError(
                "No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0:
            return WankePVP.VERBALIZER_A[label]
        else:
            raise NotImplementedError
