# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for MiniCPMV with Nicheformer integration.
"""

from typing import List, Optional, Union
import torch
import re
from PIL import Image
import anndata as ad
import numpy as np
import os
from transformers import AutoTokenizer
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType

from .image_processing_minicpmv import MiniCPMVBatchFeature
from .tokenization_nicheformer import NicheformerTokenizer


class MiniCPMVProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "gene_tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    gene_tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, gene_tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer, gene_tokenizer, **kwargs)
        self.version = kwargs.get("version", 2.6)
        # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # tokenizer_dir = os.path.join(BASE_DIR, "gene_tokenizer")
        # self.gene_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        # technology_mean_path = os.path.join(tokenizer_dir, "xenium_mean_script.npy")
        self.gene_tokenizer = AutoTokenizer.from_pretrained("/data2/xiaoxinyu/project/model/gene_tokenizer", trust_remote_code=True)
        technology_mean_path = '/data2/xiaoxinyu/project/model/gene_tokenizer/xenium_mean_script.npy'
        technology_mean = np.load(technology_mean_path)
        self.gene_tokenizer._load_technology_mean(technology_mean)

        # print(f"[DEBUG] tokenizer type = {type(self.tokenizer)}")
        # print(f"[DEBUG] gene_tokenizer type = {type(self.gene_tokenizer)}")

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        gene_data: Union[ad.AnnData, np.ndarray, List] = None,
        max_length: Optional[int] = None,
        do_pad: Optional[bool] = True,
        max_slice_nums: int = None,
        use_image_id: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs,
    ) -> MiniCPMVBatchFeature:

        # Step 1: Process images
        image_inputs = None
        if images is not None and any(img is not None for img in images):
            image_inputs = self.image_processor(
                images, do_pad=do_pad, max_slice_nums=max_slice_nums, return_tensors=return_tensors
            )
            # print(f"[DEBUG] 成功获取image_inputs : {image_inputs.keys()}")


        # Step 2: Process gene data
        gene_inputs = None
        if gene_data and len(gene_data) > 0 and len(gene_data[0]) > 0:
                adata = gene_data[0][0]
                gene_arrays = adata.X 
                gene_inputs = self.gene_tokenizer(gene_arrays)
                # print(f"[DEBUG] 成功获取gene_inputs : {gene_inputs.keys()}")


        # Step 3: Merge modalities
        return self._convert_all_modalities_to_inputs(
            image_inputs=image_inputs,
            gene_inputs=gene_inputs,
            texts=text,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            max_length=max_length,
            **kwargs,
        )

    def _convert_all_modalities_to_inputs(
        self,
        image_inputs,
        gene_inputs,
        texts: Union[str, List[str]],
        truncation=None,
        max_length=None,
        max_slice_nums=None,
        use_image_id=None,
        return_tensors=TensorType.PYTORCH,
        **kwargs,
    ):

        if isinstance(texts, str):
            texts = [texts]

        input_ids_list = []
        image_bounds_list = []
        gene_bounds_list = []

        image_pattern = "(<image>./</image>)"
        gene_pattern = "(<gene>./</gene>)"

        for index, text in enumerate(texts):
            
            image_tags = re.findall(image_pattern, text)
            if image_inputs is not None:
                image_sizes = image_inputs["image_sizes"]
                assert len(image_tags) == len(
                    image_sizes[index]
                ), f"Mismatch between image tags ({len(image_tags)}) and actual images ({len(image_sizes[index])})"

            # replace placeholders
            final_text = text
            if image_inputs is not None:
                text_chunks = final_text.split(image_pattern)
                final_text = ""
                for i in range(len(image_tags)):
                    final_text += text_chunks[i] + self.image_processor.get_slice_image_placeholder(
                        image_sizes[index][i], i, max_slice_nums, use_image_id
                    )
                final_text += text_chunks[-1]

            # === 处理 gene ===
            gene_tags = re.findall(gene_pattern, final_text)
            if gene_inputs is not None:
                text_chunks = re.split(gene_pattern, final_text)
                final_text = ""
                for i in range(len(gene_tags)):
                    gene_tokens = gene_inputs["input_ids"][index]
                    # gene_token_str = " ".join(map(str, gene_tokens.tolist()))
                    # final_text += text_chunks[i] + f"<gene_id>{i}</gene_id><gene>{gene_token_str}</gene>"
                    dummy_placeholder = "<unk>" * 32
                    final_text += text_chunks[i] + f"<gene_id>{i}</gene_id><gene>{dummy_placeholder}</gene>"
                final_text += text_chunks[-1]

            # print(f"[DeBUG] final_text: {final_text}")


            # 🔑 get input_ids and image_bounds directly
            input_ids, image_bounds, gene_bounds = self._convert(final_text, max_length)
            

            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)  # ✅ keep tensor
            gene_bounds_list.append(gene_bounds)

        # print(f"[DeBUG] input_ids: {input_ids_list}")
        # print(f"[DeBUG] input_ids length: {input_ids.size(0)}")
        # print(f"[DeBUG] image_bound: {image_bounds_list}")
        # print(f"[DeBUG] gene_bound: {gene_bounds_list}")

        # pad input_ids
        padded_input_ids, padding_lengths = self.pad(input_ids_list, padding_side="left")

        # shift bounds for padding
        for i, length in enumerate(padding_lengths):
            if image_bounds_list[i].numel() > 0:
                image_bounds_list[i] = image_bounds_list[i] + length
            if gene_bounds_list[i].numel() > 0:
                gene_bounds_list[i] = gene_bounds_list[i] + length

        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)
        
        labels = padded_input_ids.clone()
        labels[~attention_mask] = -100  # padding 不算loss
        # gene span 不算loss
        for i, gb in enumerate(gene_bounds_list):
            if torch.is_tensor(gb) and gb.numel() > 0:
                for (s, e) in gb.tolist():
                    labels[i, s:e] = -100
        
        # print(f"[DeBUG] padded_input_ids: {padded_input_ids}")
        # print(f"[DeBUG] attention_mask: {attention_mask}")
        # print(f"[DeBUG] image_bounds_list: {image_bounds_list}")
        # print(f"[DeBUG] gene_bounds_list: {gene_bounds_list}")
        
        data = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_bound": image_bounds_list,  # ✅ tensor [N,2]
            "gene_bound": gene_bounds_list,
        }

        if image_inputs:
            data.update(
                {
                    "pixel_values": image_inputs["pixel_values"],
                    "image_sizes": image_inputs["image_sizes"],
                    "tgt_sizes": image_inputs["tgt_sizes"],
                }
            )
        if gene_inputs:
            data.update(
                {
                    "gene_input_ids": gene_inputs["input_ids"],
                    "gene_attention_mask": gene_inputs["attention_mask"],
                 }
            )

        return MiniCPMVBatchFeature(data=data)

    def _convert(self, input_str, max_inp_length: Optional[int] = None):
        if self.version > 2.5 or not getattr(self.tokenizer, "add_bos_token", False):
            input_ids = self.tokenizer.encode(input_str)
        else:
            input_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(input_str)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        
         # 找 image 边界
        image_start_tokens = torch.where(
            (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        )[0] + 1
        image_end_tokens = torch.where(
            (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)
        )[0]
        valid_image_nums = min(len(image_start_tokens), len(image_end_tokens))
        image_bounds = torch.stack(
            [image_start_tokens[:valid_image_nums], image_end_tokens[:valid_image_nums]], dim=1
        )
        
        # 找 gene 边界
        gene_start_tokens = torch.where(input_ids == self.tokenizer.gene_start_id)[0] + 1
        gene_end_tokens   = torch.where(input_ids == self.tokenizer.gene_end_id)[0]
        valid_gene_nums = min(len(gene_start_tokens), len(gene_end_tokens))
        gene_bounds = torch.stack(
            [gene_start_tokens[:valid_gene_nums], gene_end_tokens[:valid_gene_nums]], dim=1
        ) if valid_gene_nums > 0 else torch.zeros((0, 2), dtype=torch.int32)
        
        # print(f"[DETAIL] self.tokenizer.gene_start_id : {self.tokenizer.gene_start_id}")
        # print(f"[DETAIL] gene_start_tokens : {gene_start_tokens}")
        # print(f"[DETAIL] self.tokenizer.gene_end_id : {self.tokenizer.gene_end_id}")
        # print(f"[DETAIL] gene_end_tokens : {gene_end_tokens}")

        return input_ids, image_bounds, gene_bounds

    def batch_decode(self, *args, **kwargs):
        output_ids = args[0]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            if result[0] == self.tokenizer.bos_id:
                result = result[1:]
            if result[-1] == self.tokenizer.eos_id:
                result = result[:-1]
            result_text.append(self.tokenizer.decode(result, *args[1:], **kwargs).strip())
        return result_text

    def decode(self, *args, **kwargs):
        result = args[0]
        result = result[result != 0]
        if result[0] == self.tokenizer.bos_id:
            result = result[1:]
        if result[-1] == self.tokenizer.eos_id or (
            hasattr(self.tokenizer, "eot_id") and result[-1] == self.tokenizer.eot_id
        ):
            result = result[:-1]
        return self.tokenizer.decode(result, *args[1:], **kwargs).strip()

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        gene_tokenizer_input_names = self.gene_tokenizer.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + gene_tokenizer_input_names))

    def pad(self, inputs, max_length=None, padding_value=0, padding_side="left"):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], torch.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], torch.Tensor)
            items = inputs

        batch_size = len(items)
        shape = items[0].shape
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item.shape[-1] for item in items))
        min_length = min(item.shape[-1] for item in items)
        dtype = items[0].dtype

        if dim == 0:
            return torch.stack([item for item in items], dim=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return torch.stack([item for item in items], dim=0), [0] * batch_size
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == "left":
                    tensor[i, -len(item) :] = item.clone()
                else:
                    tensor[i, : len(item)] = item.clone()
            elif dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item) :, :] = item.clone()
                else:
                    tensor[i, : len(item), :] = item.clone()
            padding_length.append(tensor.shape[-1] - len(item))

        return tensor, padding_length
