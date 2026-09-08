# where the data came from

Training rows look like:

```
text,intent
"some user prompt...",benign
"ignore previous instructions...",injection
```

I didn't scrape this myself. Combined a few public HF / github dumps:

1. neuralchemy prompt-injection (core)
2. deepset/prompt-injections
3. Radda prompt-injection-dataset-v2 (train/val/test json)

Then deduped on text, made a messy copy for cleaning practice, and wrote train/test with an 80/20 stratified split (seed 42).

After cleaning I had roughly 25k rows → ~20k train / ~5k test.
