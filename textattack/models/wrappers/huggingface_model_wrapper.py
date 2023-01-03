"""
HuggingFace Model Wrapper
--------------------------
"""

import tempfile
import torch
import transformers
import lightseq.inference as lsi

import textattack
from textattack.models.helpers import T5ForTextToText
from textattack.models.helpers import extract_bert_weights
from textattack.models.tokenizers import T5Tokenizer

from .pytorch_model_wrapper import PyTorchModelWrapper

torch.cuda.empty_cache()


class LightseqBertClassification:
    def __init__(self, ls_weight_path, hf_model):
        self.ls_bert = lsi.Bert(ls_weight_path, 128)
        self.hf_model = hf_model
        self.hf_model.to('cuda:0')
        self.pooler = hf_model.bert.pooler
        self.classifier = hf_model.classifier

    def infer(self, inputs, attn_mask):
        self.hf_model.eval()
        with torch.no_grad():
            last_hidden_states = self.ls_bert.infer(inputs, attn_mask)
            last_hidden_states = torch.Tensor(last_hidden_states).float()
            pooled_output = self.pooler(last_hidden_states.to(self.hf_model.device))
            logits = self.classifier(pooled_output)
        return logits


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.ls_model_name = None
        self.ls_model = None

    def _preprocess(self, text_input_list):
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        if self.ls_model is None:
            model_device = next(self.model.parameters()).device
            inputs_dict.to(model_device)
        return inputs_dict

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        inputs_dict = self._preprocess(text_input_list)

        if self.ls_model is not None:
            return self.ls_infer(inputs_dict['input_ids'], inputs_dict['attention_mask'])

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits
    
    def export_ls_model(self):
        """export Huggingface models to LightSeq models.
        """
        self.ls_model_path = tempfile.mktemp() + '.hdf5'
        extract_bert_weights(self.model, self.ls_model_path, 12)
        self.ls_model = LightseqBertClassification(self.ls_model_path, self.model)

    def ls_infer(self, input_id, attn_mask):
        """passes inputs to LightSeq models as keyword aruguments.
        """
        return self.ls_model.infer(input_id, attn_mask)

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
