import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize data storage
data = defaultdict(list)
steps = defaultdict(list)

# Path to your events file
events_file = "./events.out.tfevents.1746935501.Mac.attlocal.net.76364.0"

# Read the events file
for e in tf.compat.v1.train.summary_iterator(events_file):
    for v in e.summary.value:
        # Extract the tag and value
        if hasattr(v, 'tag') and hasattr(v, 'simple_value'):
            data[v.tag].append(v.simple_value)
            steps[v.tag].append(e.step)

# Plot the curves
plt.figure(figsize=(10, 6))
for tag in data:
    plt.plot(steps[tag], data[tag], label=tag)

plt.xlabel('Training Steps')
plt.ylabel('Loss Value')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('training_curves.png', dpi=300)
print("Plot saved as training_curves.png") 