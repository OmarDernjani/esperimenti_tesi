"""No-download CPU integration check for DPO and frozen SFT reference adapters."""
import tempfile
from pathlib import Path


def main():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import DPOConfig
    from .train import trainer_class
    torch.set_num_threads(1)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {'<unk>': 0, '<pad>': 1, '<eos>': 2, 'a': 3, 'b': 4}, unk_token='<unk>')),
        unk_token='<unk>', pad_token='<pad>', eos_token='<eos>')
    rows = Dataset.from_list([{'prompt_input_ids': [3, 4], 'chosen_input_ids': [3, 2],
                               'rejected_input_ids': [4, 2]}] * 2)
    with tempfile.TemporaryDirectory() as temp:
        for warm in (False, True):
            base = GPT2LMHeadModel(GPT2Config(vocab_size=5, n_positions=32, n_embd=16,
                n_layer=1, n_head=2, bos_token_id=2, eos_token_id=2, pad_token_id=1))
            lora = LoraConfig(r=2, lora_alpha=4, target_modules=['c_attn'], task_type='CAUSAL_LM')
            names = {}
            if warm:
                initial = Path(temp) / 'initial'
                get_peft_model(base, lora).save_pretrained(initial)
                base = GPT2LMHeadModel(base.config)
                base = PeftModel.from_pretrained(base, initial, is_trainable=True)
                base.load_adapter(initial, adapter_name='reference', is_trainable=False)
                base.set_adapter('default')
                names = dict(model_adapter_name='default', ref_adapter_name='reference')
            args = DPOConfig(output_dir=str(Path(temp) / str(warm)), use_cpu=True, bf16=False,
                max_steps=1, learning_rate=0.01, per_device_train_batch_size=1,
                precompute_ref_log_probs=True, max_length=32, max_prompt_length=None,
                save_strategy='no', report_to='none', **names)
            trainer = trainer_class()(model=base, peft_config=None if warm else lora,
                args=args, processing_class=tokenizer, train_dataset=rows, eval_dataset=rows)
            reference = {k: v.detach().clone() for k, v in trainer.model.named_parameters() if '.reference.' in k}
            policy = {k: v.detach().clone() for k, v in trainer.model.named_parameters() if v.requires_grad}
            trainer.evaluate()
            result = trainer.train()
            assert torch.isfinite(torch.tensor(result.training_loss)), 'Nonfinite loss'
            current = dict(trainer.model.named_parameters())
            assert any(not torch.equal(v, current[k]) for k, v in policy.items()), 'Policy did not update'
            assert all(torch.equal(v, current[k]) for k, v in reference.items()), 'Reference changed'
            destination = Path(temp) / f'final-{warm}'
            trainer.model.save_pretrained(destination, selected_adapters=['default'])
            assert (destination / 'adapter_model.safetensors').is_file()
            assert not (destination / 'reference').exists()
    print('PASS: CPU DPO updates policy, keeps reference fixed, saves only policy; no downloads.')


if __name__ == '__main__':
    main()
