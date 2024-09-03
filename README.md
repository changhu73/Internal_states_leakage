# LLM copyright

"git clone https://github.com/chentong0/copy-bench.git" under "copyright" folder

1. Generate output(generate_output.py)
2. Outputs clean, just keep the outputs after input sentences(outputs_clean.py)
3. Evaluate the label of output(evaluate_label.py)
4. Abstract the labels of reference, all of them are "refuse 0", the same process to labels of output(reference_label.py & output_label.py)
5. Train the MLP using the dataset above(MLPtrain.py)
