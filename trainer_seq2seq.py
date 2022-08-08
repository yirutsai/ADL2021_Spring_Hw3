# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lib2to3.pgen2.token import OP
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

import nltk
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput
from transformers.debug_utils import DebugOption
from transformers.utils import logging
from transformers.file_utils import is_torch_tpu_available
from typer import Option

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

logger = logging.get_logger(__name__)


class Seq2SeqTrainer(Trainer):
    def __init__(self,*args,top_k=None,top_p = None,temperature = None,do_sample=None,rl_ratio=None,max_length=None,num_beams=None,rl_type=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.do_sample = do_sample
        self.rl_ratio = rl_ratio
        self.max_length = max_length 
        self.rl_type = rl_type
        print(f"using num_beams:{num_beams},max_length:{max_length} top_k:{top_k}, top_p:{top_p}, temperature:{temperature}, do_sample:{do_sample},rl_ratio:{rl_ratio}")

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # self._max_length = self.max_length
        # self._num_beams = self.num_beams
        ##########original trainer
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # start_time = time.time()

        if(self.args.use_legacy_prediction_loop):
            print("using self.prediction_loop")
        else:
            print("using self.evaluation_loop")
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only= None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        # total_batch_size = self.args.eval_batch_size * self.args.world_size
        # output.metrics.update(
        #     speed_metrics(
        #         metric_key_prefix,
        #         start_time,
        #         num_samples=output.num_samples,
        #         num_steps=math.ceil(output.num_samples / total_batch_size),
        #     )
        # )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
        <Tip>
        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # self._max_length = self.max_length
        # self._num_beams = self.num_beams
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature" : self.temperature,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        from tw_rouge import get_rouge
        # print(inputs.keys())
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
            labels_copy = labels.clone()
            if(self.rl_ratio!= None):
                inputs = self._prepare_inputs(inputs)
                # XXX: adapt synced_gpus for fairscale as well
                gen_kwargs = {
                    "max_length": self.max_length,
                    "num_beams": self.num_beams,
                    "do_sample": self.do_sample,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "temperature" : self.temperature,
                    "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
                }
                if "attention_mask" in inputs:
                    gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
                if "global_attention_mask" in inputs:
                    gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

                # prepare generation inputs
                # some encoder-decoder models can have varying encoder's and thus
                # varying model input names
                if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
                    generation_inputs = inputs[self.model.encoder.main_input_name]
                else:
                    generation_inputs = inputs[self.model.main_input_name]

                greedy_tokens = self.model.generate(
                    generation_inputs,
                    num_beams =1,
                    max_length=self.max_length,
                    do_sample = False,
                    attention_mask = inputs.get("attention_mask",None)
                )
                if(self.rl_type=="sample"):
                    generated_tokens = self.model.generate(
                        generation_inputs,
                        **gen_kwargs,
                    )
                def postprocess_text(greedy_preds, preds, labels):
                    greedy_preds = [pred.strip() for pred in greedy_preds]
                    if(preds!=None):
                        preds = [pred.strip() for pred in preds]
                    labels = [label.strip() for label in labels]

                    # rougeLSum expects newline after each sentence
                    greedy_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in greedy_preds]
                    if(preds!=None):
                        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
                    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

                    return greedy_preds,preds, labels
                if(greedy_tokens.shape[-1]<gen_kwargs["max_length"]):
                    greedy_tokens = self._pad_tensors_to_max_len(greedy_tokens, gen_kwargs["max_length"])
                if(self.rl_type=="sample"):
                    if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                        generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

                decoded_greedy_preds = self.tokenizer.batch_decode(greedy_tokens,skip_special_tokens=True)
                if(self.rl_type=="sample"):
                    decoded_preds = self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
                labels = np.where(labels.cpu()!=-100,labels.cpu(),self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(labels,skip_special_tokens=True)
                if(self.rl_type=="sample"):
                    decoded_greedy_preds,decoded_preds,decoded_labels = postprocess_text(decoded_greedy_preds,decoded_preds,decoded_labels)
                else:
                    decoded_greedy_preds,_,decoded_labels = postprocess_text(decoded_greedy_preds,None,decoded_labels)
                

                decoded_greedy_preds = [pred.strip()+"\n" for pred in decoded_greedy_preds]
                if(self.rl_type=="sample"):
                    decoded_preds = [pred.strip()+"\n" for pred in decoded_preds]
                decoded_labels = [label.strip()+"\n" for label in decoded_labels]
                greedy_result = get_rouge(decoded_greedy_preds,decoded_labels)
                if(self.rl_type=="sample"):
                    myresult = get_rouge(decoded_preds,decoded_labels)
                # print("Doing RL",myresult)
                greedy_reward = greedy_result["rouge-1"]["f"]+greedy_result["rouge-2"]["f"]+greedy_result["rouge-l"]["f"]
                if(self.rl_type=="sample"):
                    reward = myresult["rouge-1"]["f"]+myresult["rouge-2"]["f"]+myresult["rouge-l"]["f"]
                    log_probs = -nn.functional.log_softmax(outputs.logits,dim = -1)
                    if(labels_copy.dim()==log_probs.dim()-1):
                        labels_copy = labels_copy.unsqueeze(-1)
                    padding_mask = labels_copy.eq(-100)
                    loss_labels = torch.clamp(labels_copy,min=0)
                    nll_loss = log_probs.gather(dim=-1,index=loss_labels)
                    nll_loss.masked_fill_(padding_mask,0.0)
                    nll_loss = nll_loss.sum()
                if(self.rl_type=="sample"):
                    loss = (1-self.rl_ratio)*loss - (torch.tensor(greedy_reward-reward)*nll_loss).mean()*self.rl_ratio
                else:
                    loss = (1-self.rl_ratio+greedy_reward*self.rl_ratio)*loss


        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss