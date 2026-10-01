import random
import os

file_all = [f for f in os.listdir('audio_files') if f.endswith('.wav')]

random.seed(42)
subset = random.sample(file_all, 30)

video_ids = [os.path.splitext(file)[0] for file in subset]
video_ids = sorted(video_ids)

print('Selected 30-video subset:')
for i, id in enumerate(video_ids, 1):
    print(f'{i}.{id}')
    
with open('selected_30_videos.txt', 'w') as f:
    for name in subset:
        f.write(name + 'n')
