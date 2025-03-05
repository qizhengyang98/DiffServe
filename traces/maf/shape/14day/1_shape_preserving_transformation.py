import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('original_invocations_per_hour.csv')

transform_min = 100
transform_max = 500
transform_difference = transform_max - transform_min

original_invocations = df['invocations'].values
plt.plot(original_invocations)
plt.xlabel('Time (hour)')
plt.ylabel('Invocations')
plt.savefig('original_invocations.png')
plt.close()

minimum = min(original_invocations)
maximum = max(original_invocations)

difference = maximum - minimum

# Starts to [0, difference]
scaled_invocations = original_invocations - minimum

# Scaled to [0, 1]
scaled_invocations /= difference

plt.plot(scaled_invocations)
plt.xlabel('Time (hour)')
plt.ylabel('Invocations')
plt.savefig('scaled_invocations.png')
plt.close()

# Scaled to [0, transform_difference]
transformed_invocations = scaled_invocations * transform_difference

# Scaled to [transform_min, transform_max]
transformed_invocations += transform_min

plt.plot(transformed_invocations)
plt.xlabel('Time (hour)')
plt.ylabel('Invocations')
plt.savefig('transformed_invocations.png')
plt.close()

transformed_df = pd.DataFrame({'hour': range(len(transformed_invocations)),
                               'invocations': transformed_invocations})
transformed_df.to_csv('transformed_invocations.csv')

plt.plot(transformed_invocations)
plt.xlabel('Time (hour)')
plt.ylabel('Invocations')
plt.ylim([0, transform_max*1.1])
plt.savefig('transformed_invocations_0axes.png')
plt.close()
