import typing as T

import hashlib
import os
import pickle as pk
import sys
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import T5EncoderModel, T5Tokenizer
from ..utils import get_logger
from .base import Featurizer

from transformers import AutoModel, AutoTokenizer, pipeline

logg = get_logger()

MODEL_CACHE_DIR = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
)
FOLDSEEK_DEFAULT_PATH = Path(
    "/afs/csail.mit.edu/u/r/rsingh/work/corals/data-scratch1/ConPLex_foldseek_embeddings/r1_foldseekrep_encoding.p"
)
FOLDSEEK_MISSING_IDX = 20

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


class BeplerBergerFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):
        super().__init__("BeplerBerger", 6165, save_dir)
        from dscript.language_model import lm_embed

        self._max_len = 800
        self._embed = lm_embed

    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[: self._max_len]

        lm_emb = self._embed(seq, use_cuda=self.on_cuda)
        return lm_emb.squeeze().mean(0)


class ESMFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):
        super().__init__("ESM", 1280, save_dir)


        import esm
        # 如果没有提供本地模型路径，则从预训练模型加载
        self._max_len = 1024
        (
                self._esm_model,
                self._esm_alphabet,
            ) = esm.pretrained.esm1b_t33_650M_UR50S()

        self._esm_batch_converter = self._esm_alphabet.get_batch_converter()
        self._register_cuda("model", self._esm_model)

    def _transform(self, seq: str):
        seq = seq.upper()
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        batch_labels, batch_strs, batch_tokens = self._esm_batch_converter(
            [("sequence", seq)]
        )
        batch_tokens = batch_tokens.to(self.device)
        results = self._cuda_registry["model"][0](
            batch_tokens, repr_layers=[33], return_contacts=True
        )
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        tokens = token_representations[0, 1 : len(seq) + 1]

        return tokens.mean(0)
    



#ProtBert特征提取器和分词器
class ProtBertFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), per_tok=False, device=None):
        super().__init__("ProtBert", 1024, save_dir)

        self._max_len = 1024
        self.per_tok = per_tok

        # 定义本地模型目录
        LOCAL_MODEL_DIR = Path("/home/liujin/data/ConPLex-main/conplex_dti/prot_bert").resolve()

        # 调试：打印解析后的路径
        print(f"解析后的probert模型目录路径: {LOCAL_MODEL_DIR}")

        # 检查目录是否存在
        if not LOCAL_MODEL_DIR.exists():
            raise FileNotFoundError(f"模型目录不存在: {LOCAL_MODEL_DIR}")

        # 检查目录内容
        print(f"目录内容: {list(LOCAL_MODEL_DIR.iterdir())}")

        self._protbert_tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_DIR,
            do_lower_case=False,
        )
        self._protbert_model = AutoModel.from_pretrained(
            LOCAL_MODEL_DIR,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用 self.__dict__ 直接设置属性
        self.__dict__['device'] = device

        # 特征提取管道
        self._protbert_feat = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
            device=device.index if device.type == "cuda" else -1,
        )

        self._register_cuda("model", self._protbert_model)
        self._register_cuda("featurizer", self._protbert_feat, self._feat_to_device)

    def _feat_to_device(self, pipe, device):
        from transformers import pipeline

        if device.type == "cpu":
            d = -1
        else:
            d = device.index

        pipe = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
            device=d,
        )
        self._protbert_feat = pipe
        return pipe

    def _space_sequence(self, x):
        return " ".join(list(x))

    def _transform(self, seq: str):
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        embedding = torch.tensor(
            self._cuda_registry["featurizer"][0](self._space_sequence(seq))
        )
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len + 1
        feats = embedding.squeeze()[start_Idx:end_Idx]

        if self.per_tok:
            return feats
        return feats.mean(0)


class ProtT5XLUniref50Featurizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), per_tok=False):
        super().__init__("ProtT5XLUniref50", 1024, save_dir)

        self._max_len = 1024
        self.per_tok = per_tok

        # 定义本地模型目录
        LOCAL_MODEL_DIR = Path("/home/liujin/data/ConPLex-main/conplex_dti/ProtT5XLUniref50").resolve()

        # 调试：打印解析后的路径
        print(f"解析后的probert模型目录路径: {LOCAL_MODEL_DIR}")

        # 检查目录是否存在
        if not LOCAL_MODEL_DIR.exists():
            raise FileNotFoundError(f"模型目录不存在: {LOCAL_MODEL_DIR}")

        # 检查目录内容
        print(f"目录内容: {list(LOCAL_MODEL_DIR.iterdir())}")

        self._protbert_tokenizer =T5Tokenizer.from_pretrained(
            LOCAL_MODEL_DIR,
            do_lower_case=False,
        )
        # T5ForConditionalGeneration
        self._protbert_model = T5EncoderModel.from_pretrained(
            LOCAL_MODEL_DIR,
        )

        # (
        #     self._protbert_model,
        #     self._protbert_tokenizer,
        # ) = ProtT5XLUniref50Featurizer._get_T5_model()
        self._register_cuda("model", self._protbert_model)

    # @staticmethod
    # def _get_T5_model():
    #     from transformers import T5EncoderModel, T5Tokenizer

    #     model = T5EncoderModel.from_pretrained(
    #         "Rostlab/prot_t5_xl_uniref50",
    #         cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers",
    #     )
    #     model = model.eval()  # set model to evaluation model
    #     tokenizer = T5Tokenizer.from_pretrained(
    #         "Rostlab/prot_t5_xl_uniref50",
    #         do_lower_case=False,
    #         cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers",
    #     )

        # return model, tokenizer

    # @staticmethod
    def _space_sequence(x):
        return " ".join(list(x))

    def _transform(self, seq: str):
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        token_encoding = self._protbert_tokenizer.batch_encode_plus(
            ProtT5XLUniref50Featurizer._space_sequence(seq),
            add_special_tokens=True,
            padding="longest",
        )
        input_ids = torch.tensor(token_encoding["input_ids"])
        attention_mask = torch.tensor(token_encoding["attention_mask"])

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            embedding = self._cuda_registry["model"][0](
                input_ids=input_ids, attention_mask=attention_mask
            )
            embedding = embedding.last_hidden_state
            seq_len = len(seq)
            start_Idx = 1
            end_Idx = seq_len + 1
            seq_emb = embedding[0][start_Idx:end_Idx]

        return seq_emb.mean(0)


