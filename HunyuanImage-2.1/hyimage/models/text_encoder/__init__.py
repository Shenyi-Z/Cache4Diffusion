from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoTokenizer

from transformers.utils import ModelOutput


def use_default(value, default):
    """Utility: return value if not None, else default."""
    return value if value is not None else default

# Prompt templates for different models and tasks
PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
PROMPT_TEMPLATE_ENCODE_V2 = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>"
)

NEGATIVE_PROMPT = (
    "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, "
    "bad hands, bad teeth, bad eyes, bad limbs, distortion"
)

PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
        "crop_start": 36,
    },
    "dit-llm-encode-v2": {
        "template": PROMPT_TEMPLATE_ENCODE_V2,
        "crop_start": 34,
    },
}

def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    infer_mode="encoder",
    logger=None,
    device=None
):
    """
    Load a text encoder model from pretrained weights.

    Args:
        text_encoder_type (str): Type of text encoder.
        text_encoder_precision (str, optional): Precision for model weights.
        text_encoder_path (str, optional): Path to pretrained weights.
        infer_mode (str): "encoder" or "decoder".
        logger (logging.Logger, optional): Logger for info.
        device (torch.device, optional): Device to move model to.

    Returns:
        model (nn.Module): Loaded text encoder.
        model_path (str): Path to model.
    """
    if logger is not None:
        logger.info(f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}")

    if text_encoder_type == 'llm':
        text_encoder = AutoModelForVision2Seq.from_pretrained(
            text_encoder_path,
            torch_dtype="auto"
        )
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path

def load_tokenizer(
    tokenizer_type,
    tokenizer_path=None,
    padding_side="right",
    logger=None
):
    """
    Load a tokenizer from pretrained weights.

    Args:
        tokenizer_type (str): Type of tokenizer.
        tokenizer_path (str, optional): Path to pretrained tokenizer.
        padding_side (str): Padding side for tokenizer.
        logger (logging.Logger, optional): Logger for info.

    Returns:
        tokenizer: Loaded tokenizer.
        tokenizer_path (str): Path to tokenizer.
    """
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type == "llm":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=False, padding_side=padding_side, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path

@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Output for text encoder models.

    Args:
        hidden_state (torch.FloatTensor): Output hidden states of the last layer.
        attention_mask (torch.LongTensor, optional): Attention mask for valid tokens.
        hidden_states_list (tuple(torch.FloatTensor), optional): All hidden states if requested.
        text_outputs (list, optional): Decoded texts if requested.
    """
    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None

class TextEncoder(nn.Module):
    """
    TextEncoder wraps a pretrained text encoder and tokenizer for flexible text encoding.

    Args:
        text_encoder_type (str): Type of text encoder.
        max_length (int): Maximum sequence length.
        text_encoder_precision (str, optional): Precision for model weights.
        text_encoder_path (str, optional): Path to pretrained weights.
        tokenizer_type (str, optional): Type of tokenizer.
        tokenizer_path (str, optional): Path to pretrained tokenizer.
        output_key (str, optional): Output key for model output.
        use_attention_mask (bool): Whether to use attention mask.
        infer_mode (str): "encoder" or "decoder".
        input_max_length (int, optional): Max input length.
        prompt_template (dict, optional): Prompt template for image.
        prompt_template_video (dict, optional): Prompt template for video.
        hidden_state_skip_layer (int, optional): Skip layers from last for hidden state.
        apply_final_norm (bool): Whether to apply final layer norm.
        reproduce (bool): Deterministic output if True.
        logger (logging.Logger, optional): Logger for info.
        device (torch.device, optional): Device to move model to.
    """
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        infer_mode: str = "encoder",
        input_max_length: Optional[int] = None,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = tokenizer_type if tokenizer_type is not None else text_encoder_type
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else text_encoder_path
        self.use_attention_mask = use_attention_mask
        self.input_max_length = input_max_length if input_max_length is not None else max_length
        self.prompt_template = dict(prompt_template) if prompt_template is not None else None
        self.prompt_template_video = dict(prompt_template_video) if prompt_template_video is not None else None
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.infer_mode = infer_mode
        self.reproduce = reproduce
        self.logger = logger

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert isinstance(self.prompt_template, dict) and "template" in self.prompt_template, (
                f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            )
            if self.prompt_template_video is not None:
                assert isinstance(self.prompt_template_video, dict) and "template" in self.prompt_template_video, (
                    f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
                )
            assert '{}' in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        if infer_mode == "decoder":
            assert text_encoder_type in ["llava-llama-3-8b"], (
                f"Unsupported text encoder type for infer_mode='decoder': {text_encoder_type}"
            )
            assert self.prompt_template is not None and hidden_state_skip_layer is not None, (
                f"`prompt_template` and `hidden_state_skip_layer` must be provided for infer_mode='decoder', "
                f"got prompt_template={self.prompt_template}, hidden_state_skip_layer={self.hidden_state_skip_layer}"
            )

        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif any(x in text_encoder_type for x in ["llm"]):
            self.output_key = output_key or ("last_hidden_state" if infer_mode == "encoder" else None)
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            infer_mode=self.infer_mode,
            logger=self.logger,
            device=device
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        padding_side = "right" if self.infer_mode == "encoder" else "left"
        self.tokenizer, self.tokenizer_path = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side=padding_side,
            logger=self.logger
        )

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to a prompt template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If True, prevent empty user text by adding a space.

        Returns:
            str or list: Text with template applied.
        """
        if isinstance(template, str):
            return template.format(text)
        elif isinstance(template, list):
            conversation = deepcopy(template)
            for message in conversation:
                if '{}' in message.get("content", ""):
                    filled_text = message["content"].format(text)
                    if prevent_empty_text and len(filled_text) == 0:
                        filled_text = ' '
                    message["content"] = filled_text
                    break  # Only one placeholder per conversation
            return conversation
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def text2tokens(self, text, data_type='image'):
        """
        Tokenize the input text, optionally applying a prompt template.

        Args:
            text (str or list): Input text.
            data_type (str): 'image' or 'video'.

        Returns:
            dict: Tokenized input.
        """
        tokenize_input_type = 'str'
        if self.use_template:
            if data_type == 'image':
                prompt_template = self.prompt_template["template"]
            elif data_type == 'video':
                prompt_template = self.prompt_template_video["template"]
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if isinstance(text, (list, tuple)):
                text = [self.apply_text_to_template(one_text, prompt_template) for one_text in text]
                if isinstance(text[0], list):
                    tokenize_input_type = 'list'
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = 'list'
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")
        kwargs = dict(truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        if tokenize_input_type == 'str':
            return self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == 'list':
            return self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type='image',
        device=None
    ):
        """
        Encode tokenized input using the text encoder.

        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool, optional): Whether to use attention mask.
            output_hidden_states (bool): Whether to output all hidden states.
            do_sample (bool, optional): Whether to sample from the model (for decoder-only LLMs).
            hidden_state_skip_layer (int, optional): Number of layers to skip from last for hidden state.
            return_texts (bool): Whether to return decoded texts.
            data_type (str): 'image' or 'video'.
            device (torch.device, optional): Device to use.

        Returns:
            TextEncoderModelOutput: Encoded output.
        """
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(hidden_state_skip_layer, self.hidden_state_skip_layer)
        do_sample = use_default(do_sample, not self.reproduce)

        if self.infer_mode == "encoder":
            attention_mask = batch_encoding["attention_mask"].to(self.model.device) if use_attention_mask else None
            if 'Gemma2' in self.text_encoder_type:
                input_ids = batch_encoding["input_ids"].to(self.model.device)
                _, inputs_embeds, labels, attention_mask = self.model.merge_multimodal(
                    text_input_ids=input_ids,
                    text_attention_masks=attention_mask,
                    text_labels=None,
                    pixel_values=[None]
                )
                outputs = self.model.llm(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)
            else:
                outputs = self.model(
                    input_ids=batch_encoding["input_ids"].to(self.model.device),
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None,
                )
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
                # Apply final norm for intermediate layers if requested
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            # Remove hidden states of instruction tokens, only keep prompt tokens.
            if self.use_template:
                if data_type == 'image':
                    crop_start = self.prompt_template.get("crop_start", -1)
                elif data_type == 'video':
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
                if crop_start > 0:
                    last_hidden_state = last_hidden_state[:, crop_start:]
                    attention_mask = attention_mask[:, crop_start:] if use_attention_mask else None

            if output_hidden_states:
                return TextEncoderModelOutput(last_hidden_state, attention_mask, outputs.hidden_states)
            return TextEncoderModelOutput(last_hidden_state, attention_mask)

        elif self.infer_mode == "decoder":
            # Remove leading padding tokens
            input_max_valid_tokens = batch_encoding["attention_mask"].sum(dim=1).max().item()
            if input_max_valid_tokens < batch_encoding["attention_mask"].shape[1]:
                batch_encoding = {
                    "input_ids": batch_encoding["input_ids"][:, -input_max_valid_tokens:],
                    "attention_mask": batch_encoding["attention_mask"][:, -input_max_valid_tokens:],
                }

            # Generate text from the model.
            outputs = self.model.generate(
                input_ids=batch_encoding["input_ids"].to(self.model.device),
                attention_mask=batch_encoding["attention_mask"].to(self.model.device) if use_attention_mask else None,
                max_new_tokens=self.max_length,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_hidden_states=True,
                stop_strings='<|eot_id|>', tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Concatenate hidden states from all generated tokens.
            hidden_states = torch.cat([
                per_token_hidden_states[-(hidden_state_skip_layer + 1)]
                for per_token_hidden_states in outputs.hidden_states[1:]
            ], dim=1)
            if self.apply_final_norm:
                hidden_states = self.model.final_layer_norm(hidden_states)

            # Make sequence mask from output sequences
            output_max_valid_tokens = hidden_states.shape[1]
            attention_mask = (outputs.sequences[:, -output_max_valid_tokens - 1:-1] != self.tokenizer.eos_token_id).long()

            if return_texts:
                text_outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
                return TextEncoderModelOutput(hidden_states, attention_mask, None, text_outputs)
            else:
                return TextEncoderModelOutput(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unsupported text encoder infer mode: {self.infer_mode}")

    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False
    ):
        """
        Forward pass: encode text to hidden states.

        Args:
            text (str or list): Input text.
            use_attention_mask (bool, optional): Whether to use attention mask.
            output_hidden_states (bool): Whether to output all hidden states.
            do_sample (bool): Whether to sample from the model (for decoder-only LLMs).
            hidden_state_skip_layer (int, optional): Number of layers to skip from last for hidden state.
            return_texts (bool): Whether to return decoded texts.

        Returns:
            TextEncoderModelOutput: Encoded output.
        """
        batch_encoding = self.text2tokens(text)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts
        )
