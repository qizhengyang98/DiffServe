# Takes in the request per second shape of the trace and generates a request
# trace from it using Poisson arrivals
import numpy as np
import pandas as pd


df = pd.read_csv('transformed_invocations.csv')
invocations = df['invocations'].values

trace = []
offset = 0
for hour in range(len(invocations)):
    requests = round(invocations[hour])
                    
    if requests == 0:
        raise Exception(f'No requests at hour: {hour}')
        continue

    # we need to distribute these requests around the entire second using the exponential
    # distribution for inter-arrival rates. In other words, the requests follow Poisson
    # arrival within the second
    arrivals = np.rint(np.random.exponential(scale=1000/requests,
                                             size=requests))
    # plt.hist(arrivals)
    # plt.savefig('exponential_arrivals.pdf')
    # plt.close()

    current = 0
    # print(arrivals)
    for time in arrivals:
        current += time
        if time < 1000:
            trace.append(offset + current)

    # at the end of the second, we increment the offset in milliseconds
    offset += 1000

with open('trace/trace.txt', mode='w') as wf:
    for arrival_time in trace:
        wf.write(str(arrival_time) + '\n')
