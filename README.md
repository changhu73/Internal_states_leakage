# LLM copyright

"git clone https://github.com/chentong0/copy-bench.git" under "copyright" folder

eg:
1. Generate output(scripts/generate.py)
2. Evaluate rouge score(scripts/eval_literal_copying.py)
3. Divide into infringement/non-infringement(scripts/label.py)
4. Extract internal states by input with labels, train customMLP by internal states(scripts/train.py)
