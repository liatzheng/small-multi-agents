# Chain-length calibration

Qwen3-4B ran on one development pair at each of two chain lengths. The problem generator used seed 2027. The single-agent runner, sampling seed 42, and 4,000-token cap match the original pilot. This check varies both the number of relevant updates and the number of distractor updates across lengths. Within each pair the two variants differ by one operand.

| Updates per chain | Problem | Expected | Parsed answer | Score | Input | Output | Total |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 2 | independent | (69, 19) | (69, 19) | `correct` | 206 | 2,816 | 3,022 |
| 2 | dependent | (69, 96) | (69, 96) | `correct` | 206 | 3,478 | 3,684 |
| 4 | independent | (18, 85) | (18, 85) | `correct` | 325 | 3,144 | 3,469 |
| 4 | dependent | (18, 52) | (18, 52) | `correct` | 325 | 3,505 | 3,830 |

The length-2 run used the default attention implementation. The first length-4 attempt hit a GPU sampling error and produced no scored row. The reported length-4 run used eager attention with the same model weights, BF16 precision, sampling seed, and token cap. Backend choice can change sampled outputs.

A length-limit outcome means the model exhausted its allowance before finishing. The calibration has one pair per length, so these observations guide the next difficulty choice and do not estimate accuracy across new pairs.

[Length-2 prompts](d2.jsonl) · [Length-2 responses](single_d2.jsonl) · [Length-4 prompts](d4.jsonl) · [Length-4 responses](single_d4.jsonl)
