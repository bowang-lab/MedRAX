from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from medrax.llava.model import LlavaMistralForCausalLM
from medrax.llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_in_8bit=False,
    load_in_4bit=True,
    device="cuda",
    cache_dir: str = "/model-weights",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
):

    kwargs = {}

    if device != "cuda":
        kwargs["device_map"] = {"": device}
    # else:
    #     kwargs["device_map"] = "auto"

    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        # kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    # else:
    # kwargs["torch_dtype"] = torch_dtype

    if "llava" in model_name.lower():
        # Load LLaVA model
        if "mistral" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
            model = LlavaMistralForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_flash_attention_2=False,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                **kwargs,
            )

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                **kwargs,
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch_dtype)
        else:
            use_fast = False
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, cache_dir=cache_dir)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=cache_dir)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )

    image_processor = None

    if "llava" in model_name.lower():  # or 'mistral' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        vision_tower.to(device=device, dtype=torch_dtype)
        model.model.mm_projector.to(device=device, dtype=torch_dtype)

        if not (load_in_4bit or load_in_8bit):
            model.to(device=device, dtype=torch_dtype)

        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
