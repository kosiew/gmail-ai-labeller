# gmail-ai-labeller

This script adds labels to mails in my gmail for easy management.

## Running Typer Commands

To run the Typer commands, use the following syntax:

```sh
python main.py COMMAND [OPTIONS]
```

### Example Commands

To list all available commands:
```sh
python main.py --help
```

To append rows from one CSV to another:
```sh
python main.py append-to-extracted-emails --input-csv INPUT_CSV --output-csv OUTPUT_CSV
```

To label emails using the default LLM classifier:
```sh
python main.py llm-label
```

To label emails using the sklearn classifier:
```sh
python main.py sklearn-label
```

To extract data from processed emails:
```sh
python main.py extract-data-from-processed-emails --output-csv OUTPUT_CSV
```

To train the sklearn model from a CSV file:
```sh
python main.py train-sklearn-model-from-csv --input-csv INPUT_CSV --model-path MODEL_PATH
```

## Update sklearn model
To update the sklearn model, collect a few days of unprocessed data, then
1. extract-data
2. label the data 
3. append the data to _extracted
4. run training