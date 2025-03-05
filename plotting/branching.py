import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_dir = 'profiling/branching'
figures_dir = 'figures/'

models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
# models = ['yolov5x', 'yolov5m', 'yolov5s', 'yolov5n']
# models = ['yolov5x', 'yolov5l', 'yolov5m', 'yolov5s', 'yolov5n']

frames_per_sec = 2
car_ymax = 0
face_ymax = 0
mp_max = 0
for model in models:
    branching_file = os.path.join(data_dir, f'branching_{model}_7120.csv')
    df = pd.read_csv(branching_file)

    if max(df['car_branch']) > car_ymax:
        car_ymax = max(df['car_branch'])
    if max(df['face_branch']) > face_ymax:
        face_ymax = max(df['face_branch'])
    
    multiplicative_factor = df['car_branch'] + df['face_branch']
    if max(multiplicative_factor) > mp_max:
        mp_max = max(multiplicative_factor)


avg_car_branch_probabilities = []
avg_multiplicative_factors = []
for model in models:
    branching_file = os.path.join(data_dir, f'branching_{model}_7120.csv')
    df = pd.read_csv(branching_file)
    
    plt.plot(df['frame'] / frames_per_sec, df['car_branch'])
    plt.xlabel('Second')
    plt.ylabel('Car branch taken')
    plt.ylim(0, (car_ymax + car_ymax*0.1))
    plt.title(f'Object detection model: {model}')
    plt.savefig(os.path.join(figures_dir, f'branching_{model}_car.pdf'))
    plt.close()

    print(f'Average branch frequency for car: {np.nanmean(df["car_branch"])}')

    plt.plot(df['frame'] / frames_per_sec, df['face_branch'])
    plt.xlabel('Second')
    plt.ylabel('Face branch taken')
    plt.ylim(0, (face_ymax + face_ymax*0.1))
    plt.title(f'Object detection model: {model}')
    plt.savefig(os.path.join(figures_dir, f'branching_{model}_face.pdf'))
    plt.close()

    multiplicative_factor = df['car_branch'] + df['face_branch']

    plt.plot(df['frame'] / frames_per_sec, multiplicative_factor)
    plt.xlabel('Second')
    plt.ylabel('Multiplicative factor')
    plt.ylim(0, (mp_max + mp_max*0.1))
    plt.title(f'Object detection model: {model}')
    plt.savefig(os.path.join(figures_dir, f'multiplicative_factor_{model}.pdf'))
    plt.close()

    avg_multiplicative_factor = np.nanmean(multiplicative_factor)
    print(f'Multiplicative factor: {avg_multiplicative_factor}')
    avg_multiplicative_factors.append(avg_multiplicative_factor)

    branching_probability = {}
    branching_probability['car_branch'] = df['car_branch'] / multiplicative_factor
    branching_probability['face_branch'] = df['face_branch']  / multiplicative_factor

    plt.plot(df['frame'] / frames_per_sec, branching_probability['car_branch'])
    plt.xlabel('Second')
    plt.ylabel('Branching probability (Car)')
    plt.title(f'Object detection model: {model}')
    plt.savefig(os.path.join(figures_dir, f'branching_prob_{model}_car.pdf'))
    plt.close()

    plt.plot(df['frame'] / frames_per_sec, branching_probability['face_branch'])
    plt.xlabel('Second')
    plt.ylabel('Branching probability (Face)')
    plt.title(f'Object detection model: {model}')
    plt.savefig(os.path.join(figures_dir, f'branching_prob_{model}_face.pdf'))
    plt.close()

    avg_car_branch_prob = np.nanmean(branching_probability['car_branch'])
    print(f'Average branch probability for car: {avg_car_branch_prob}')
    avg_car_branch_probabilities.append(avg_car_branch_prob)

    print(f'Plotting done for model: {model}')

fig = plt.figure(figsize=(5, 4))
plt.plot(models, avg_car_branch_probabilities, marker='.', markersize=7)
plt.xlabel('Model')
plt.ylabel('Average branch probability (Car)')
plt.ylim(0, 1)
plt.grid()
fig.tight_layout()
plt.savefig(os.path.join(figures_dir, f'avg_car_branch_probs.pdf'))
plt.close()

fig = plt.figure(figsize=(5, 4))
plt.plot(models, avg_multiplicative_factors, marker='.', markersize=7)
plt.xlabel('Model')
plt.ylabel('Average multiplicative factor')
plt.ylim(0, max(avg_multiplicative_factors)*1.1)
plt.grid()
fig.tight_layout()
plt.savefig(os.path.join(figures_dir, f'avg_multiplicative_factors.pdf'))
plt.close()

columns = ['model', 'avg_mult_factor', 'avg_car_branch_prob', 'avg_face_branch_prob']
df = pd.DataFrame(columns=columns)
for idx in range(len(models)):
    model = models[idx]
    avg_mult_factor = avg_multiplicative_factors[idx]
    avg_car_branch_prob = avg_car_branch_probabilities[idx]
    avg_face_branch_prob = 1 - avg_car_branch_prob

    new_row = {'model': model, 'avg_mult_factor': avg_mult_factor,
               'avg_car_branch_prob': avg_car_branch_prob,
               'avg_face_branch_prob': avg_face_branch_prob}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

print(df)
df.to_csv(os.path.join('profiling/profiled', 'branching.csv'))
