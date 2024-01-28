import pandas as pd
import matplotlib.pyplot as plt

# short example of how to view the exported metadata
# shows nuclei centers and the averaged nuclei velocity
# (velocity is scaled by resolution dependet value
#  for simplicity here a constant is assumed that 
#  would have to be hand tuned)

df = pd.read_csv('metadata.csv')

grouped = df.groupby('frame_id')

for frame_id, frame_data in grouped:
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    
    plt.scatter(frame_data['center_x[px]'], frame_data['center_y[px]'], label='Center Positions', marker='o')
    
    velocity_x = 2000*frame_data['vx[mu/s]']
    velocity_y = 2000*frame_data['vy[mu/s]']
    plt.quiver(
        frame_data['center_x[px]'], frame_data['center_y[px]'],
        velocity_x, velocity_y, angles='xy', scale_units='xy', scale=1, color='red', label='Velocity Vectors'
    )
    
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title(f'Frame {frame_id}')
    plt.legend()
    
    plt.show()
