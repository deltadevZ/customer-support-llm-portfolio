# Findings

## Research Question

Can a Mistral-7B-Instruct model be adapted with QLoRA to produce more relevant customer support responses than the base zero-shot model, while keeping training hardware requirements manageable?

## Approach

The project used the Bitext customer support dataset, which contains customer questions mapped to support-style responses across multiple intents and categories. The data was formatted into an instruction-following prompt style, then used to fine-tune Mistral-7B-Instruct with LoRA adapters under 4-bit quantization.

The evaluation compared the fine-tuned model with a zero-shot baseline using automatic metrics, qualitative examples, and intent-level breakdowns.

## Key Results

| Metric | Fine-tuned | Zero-shot | Change |
| --- | ---: | ---: | ---: |
| ROUGE-1 | 0.4844 | 0.3976 | +21.8% |
| ROUGE-2 | 0.2367 | 0.1337 | +77.0% |
| ROUGE-L | 0.3368 | 0.2347 | +43.5% |
| Perplexity | 2.15 | 9.21 | -76.7% |

The fine-tuned model showed a clear improvement over the baseline on lexical overlap metrics and perplexity. This suggests that even a relatively small adapter can meaningfully adapt a general instruction model to a narrower customer support response style.

## Interpretation

The largest relative gain was on ROUGE-2, which points to better phrase-level similarity with the expected support answers. Perplexity also dropped substantially, indicating that the fine-tuned model assigned much higher probability to the target response style.

The project also included per-intent analysis. This is important because aggregate metrics can hide weak categories. Some intents performed better than others, suggesting that future improvements should focus on intent balance, harder examples, and more robust evaluation beyond average ROUGE.

## Limitations

- ROUGE and BLEU are useful but limited for customer support quality because they reward text overlap rather than factual correctness, helpfulness, or policy compliance.
- The model was evaluated on sampled subsets, so results should be treated as indicative rather than final benchmark numbers.
- The multi-turn evaluation showed weaker context/entity retention, meaning the model was better at single-turn support responses than sustained conversations.
- The adapter depends on the base model license and requires GPU resources for reproduction.

## What I Would Improve Next

- Add human evaluation criteria such as helpfulness, correctness, tone, and escalation behavior.
- Improve multi-turn training examples to strengthen context retention.
- Add safety and refusal tests for support scenarios involving personal data or account access.
- Compare multiple adapter ranks and training lengths more systematically.
- Package the adapter on Hugging Face if licensing and size constraints allow it.

## Portfolio Takeaway

This project demonstrates the full applied LLM workflow: dataset preparation, parameter-efficient fine-tuning, quantitative evaluation, qualitative error analysis, and API/demo integration.
