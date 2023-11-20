from json import dump
from pathlib import Path

from transformers import RobertaTokenizerFast

if __name__ == "__main__":
    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained("roberta-base")
    vocab_dir = Path("data") / "vocab"
    tokenizer.add_tokens([line.strip() for line in (vocab_dir / "new_vocab.txt").open()])
    tokenizer.save_pretrained("./tokenizer")

    with (vocab_dir / "tokens.json").open("w", encoding="utf8") as f:
        dump(
            {
                "vocab": len(tokenizer),
                "bos": tokenizer.bos_token_id,
                "eos": tokenizer.eos_token_id,
                "mask": tokenizer.mask_token_id,
                "top": tokenizer.convert_tokens_to_ids("multi-sentence"),
                "top_down": tokenizer.convert_tokens_to_ids(":snt1"),
                "top_up": tokenizer.convert_tokens_to_ids(":snt1-of"),
            },
            f,
        )
