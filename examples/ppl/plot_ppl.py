import matplotlib.pyplot as plt
import numpy as np

# Paths to the log files
log_files = {
    'Full': 'results/ppl/full/full_log.txt',
    # 'TD+L9(ours)': 'results/ppl/2048_9/index_9_log.txt',
    # 'TD+L13(ours)': 'results/ppl/2048_13/index_13_log.txt',
    # 'TD+L15(ours)': 'results/ppl/2048_15/index_15_log.txt',
    # 'Quest': 'results/ppl/2048_quest/quest_log.txt',
    'TD+L9(ours)': 'results/ppl/4096_9/log.txt',
    'TD+L13(ours)': 'results/ppl/4096_13/log.txt',
    'TD+L15(ours)': 'results/ppl/4096_15/log.txt',
    'Quest': 'results/ppl/4096_quest/log.txt',
}
colors = {
    'Full': 'black',
    'Quest': 'blue',
    'TD+L9(ours)': 'red',
    'TD+L13(ours)': 'purple',
    'TD+L15(ours)': 'green',
}

line_styles = {
    'Full': 'dashed',  # Use dashed line for 'Full'
    'Quest': 'solid',   # Solid line for others
    'TD+L9(ours)': 'solid',
    'TD+L13(ours)': 'solid',
    'TD+L15(ours)': 'solid',
}


# Function to read the NLL values from the log file and compute perplexity step by step
def read_nll_values_and_compute_perplexity(file_path):
    cumulative_nll = 0.0  # Cumulative sum of NLL values
    perplexities = []
    
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                nll = float(line.strip())
                cumulative_nll += nll  # Accumulate the NLL values
                
                # Compute the average NLL so far
                avg_nll = cumulative_nll / (idx + 1)
                
                # Compute perplexity as exp of the average NLL
                ppl = np.exp(avg_nll)
                perplexities.append(ppl)
            except ValueError:
                pass  # Ignore lines that cannot be parsed as float
    
    return perplexities

# Reading NLL and computing perplexities for each approach
perplexities = {label: read_nll_values_and_compute_perplexity(file) for label, file in log_files.items()}

# Increase the font sizes globally for all plot elements
plt.rcParams.update({'font.size': 18})  # Adjust the font size as needed

# Plotting perplexities for all approaches on the same graph with limited y-axis range
plt.figure(figsize=(10, 6))

for label, perp in perplexities.items():
    if label == 'Full':
        # Apply thinner, finer dashed line
        plt.plot(perp, label=label, color=colors[label], linestyle='--', linewidth=1.5, dashes=(5, 5), markersize=4)  # Thinner dashes + small circle markers
    else:
        plt.plot(perp, label=label, color=colors[label], linestyle='solid')
    print(f"Final perplexity for {label}: {perp[-1]}")

# Manually adjust individual font sizes
plt.xlabel("Input Context Length", fontsize=20)
plt.ylabel("Perplexity (the lower the better)", fontsize=20)
plt.title("Perplexity with Context Length", fontsize=20)
plt.ylim(7, 9.5)  # Limit the y-axis range between 7 and 9.5

# Reordering the legend and improving "Full"
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 4, 3, 2, 1]  # Indexes of the new order: Full, Quest, TD+L15, TD+L13, TD+L9
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=16, fancybox=True, shadow=True)

# Save the plot
plt.savefig('results/ppl/perplexity_2048.pdf', dpi=500, bbox_inches="tight")

# Show the plot
plt.show()