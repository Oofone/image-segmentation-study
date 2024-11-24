from transformers import SegformerImageProcessor, AutoImageProcessor, SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput
from peft import LoraConfig, get_peft_model

from typing import Literal, Tuple, Optional, Union, List, get_args
from torch import FloatTensor, Tensor, nn


_SegformerTrainMethod = Literal[
    "full_train",
    "encoder_lora_decode_head_full",
    "decode_head_full_only",
]
ALLOWED_SEGFORMER_TRAIN_METHODS: Tuple[_SegformerTrainMethod, ...] = get_args(_SegformerTrainMethod)

class HFSegformerModule(nn.Module):

    def __init__(
            self, n_class: int, segformer_train_method: _SegformerTrainMethod, d_model: int = 256,
            hf_model_checkpoint_path: str="nvidia/segformer-b0-finetuned-ade-512-512",
            lora_rank: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1,
            full_lora_target_modules: List[str] = ["query", "key", "value"],
            **kwargs) -> None:
        super().__init__()

        assert segformer_train_method in ALLOWED_SEGFORMER_TRAIN_METHODS, f"Train method [{segformer_train_method}] unsupported; Should be one of {ALLOWED_SEGFORMER_TRAIN_METHODS}"
        self.image_processor: SegformerImageProcessor = AutoImageProcessor.from_pretrained(hf_model_checkpoint_path)
        self.model: SegformerForSemanticSegmentation = SegformerForSemanticSegmentation.from_pretrained(hf_model_checkpoint_path)
        self.model.decode_head.classifier = nn.Conv2d(d_model, n_class, kernel_size=(1, 1), stride=(1, 1))

        if segformer_train_method == "encoder_lora_decode_head_full":
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=full_lora_target_modules,
                lora_dropout=lora_dropout,
                bias="lora_only",
                modules_to_save=["decode_head"])
            self.model = get_peft_model(model=self.model, peft_config=lora_config)
        elif segformer_train_method == "decode_head_full_only":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.decode_head.parameters():
                param.requires_grad = True

    def forward(self, X: Tensor, Y: Optional[Tensor] = None) -> Tuple[Union[Tensor, FloatTensor, None]]:
        device = X.device
        X = self.image_processor(X, return_tensors='pt', do_rescale=False)['pixel_values'].to(device)
        if Y is not None:
            output: SemanticSegmenterOutput = self.model(pixel_values=X, labels=Y.long())
            return (output.logits, output.loss)
        else:
            return (self.model(pixel_values=X, labels=None).logits, None)
