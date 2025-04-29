# CombiBench

<p align="center">
    <a href="https://huggingface.co/datasets/AI-MO/CombiBench"><img src="https://img.shields.io/badge/🤗-huggingface-FFD21E"></a>
    <a href="https://moonshotai.github.io/CombiBench/"><img src="https://img.shields.io/badge/%F0%9F%A4%96-website-87CEEB"></a>
    <a href="https://moonshotai.github.io/CombiBench/leaderboard.html"><img src="https://img.shields.io/badge/🏆-leaderboard-%23ff8811"></a>
</p>

CombiBench is the first benchmark focused on combinatorial problems, based on the formal language LEAN4. CombiBench is a manually produced benchmark, including 100 combinatorial mathematics problems of varying difficulty and knowledge levels. It aims to provide a benchmark for evaluating the combinatorial mathematics capabilities of automated theorem proving systems to advance the field. For problems that require providing a solution first and then proving its correctness, we have referred to the style of PutnamBench.

We are hosting a [**leaderboard**](https://moonshotai.github.io/CombiBench/leaderboard.html) and will readily receive evaluation results which are accompanied by a preprint or publication. Please reach out privately at `liujunqi@amss.ac.cn` with any requests for additions to the leaderboard. 

## Statistics 

We collected all combinatorics problems from the official IMO problems since 2000, except for one problem that relies on a figure. And We selected problems through random sampling from 14 chapters in the book, choosing 3 problems from each chapter, ensuring that the 42 problems are evenly distributed across all 14 chapters. We randomly selected 10 simple combinatorics problems at the middle school level from a mathematics problem collection website [hackmath](https://www.hackmath.net/). Then, we randomly collected 12 problems from other mathematics competitions.

| Source           | Count          | 
| ---------------- | -------------- | 
| Easy             | 10             |
| Brualdi's book   | 42             |
| IMO              | 36             |
| APMO             | 2              |
| Balticway        | 1              |
| EGMO             | 1              |
| IMO-Shortlist    | 4              |
| IZHO             | 2              |
| BXMO             | 1              |
| USAMO            | 1              |


## Requirements

This project requires `python >= 3.10`, `Lean>=4.15.0`.

Install the Python dependencies:

```
pip install -e .
```

or

```
uv venv .venv --prompt combibench
source .venv/bin/activate
uv sync
```

## Usage
### Setup a Lean Server

Follow https://github.com/project-numina/kimina-lean-server to configure a lean server and get a custom url and API_KEY.

### Setup a LLM API Key

We support API interfaces such as OpenAI, Antropic, TogetherAI, and Google GenerativeAI.

### Configuration

Refer to `evaluation/config/template.json5` to configure the dataset, lean server, llm server, generation parameters, prompt, and parallel parameters.

### Run Evaluation

To run one-stage Fine-Eval:

```
python evaluation/online_one_stage.py --config evaluation/config/template.json5
```

To run two-stage Fine-Eval:

```
python evaluation/online_two_stage.py --config evaluation/config/template.json5
```

Note that both evaluation methods are compatible with theorem proving tasks and fill-in-the-blank tasks.

## 🙌 Contributing

Contributions are welcome! If anyone notices any mistakes, please raise an issue on the repository and we will address it.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for full details.