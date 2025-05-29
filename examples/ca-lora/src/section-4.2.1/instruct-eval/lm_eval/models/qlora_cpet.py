import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lm_eval.base import BaseLM
from loras_f import LoraModel as LoraModelQlora


class QloraCA-LoRALM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="huggyllama/llama-7b",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_8bit=True,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)
        self.batch_size_per_gpu = batch_size

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")


        model_name = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
        tok_path = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
        adapters_name = 'timdettmers/qlora-alpaca-13b'

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.float32,
            device_map="auto",
            max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, unk_token ="<s>")

        self.vocab_size = len(self.tokenizer)

        delta_model = LoraModelQlora(
                backbone_model=self.model,
                modified_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                lora_r=16,
                backend='hf',
                lora_type='full',
        )
        delta2_model = LoraModelQlora(
                backbone_model=self.model,
                modified_modules=['gate_proj', 'up_proj', 'down_proj'],
                lora_r = 16, # TODO
                lora_dropout=0.05,
                backend='hf',
                lora_type='activate',
        )
        ckpt = torch.load('/cdgm0705/hyx/lora-thin-2000.pt', map_location=lambda storage, loc: storage.cuda(0))
        self.model.load_state_dict(ckpt, strict=False)
        delta_model.log()
        self.model.eval()


    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
