import av
import os
import numpy as np
import matplotlib.pyplot as plt


def process_video(video_path, dfs, interval=30):
    output_path = '/'.join(video_path.split('/')[:-1]) + '/pics/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the video file
    container = av.open(video_path)

    # Iterate over packets in the container
    for i, packet in enumerate(container.demux()):
        # Decode each frame in the packet
        for frame in packet.decode():
            if i % interval == 0:
                # Convert the frame to an image and then to a numpy array
                img = frame.to_image()  # PIL/Pillow image
                arr = np.asarray(img)

                # Process the frame
                fig = plt.figure(figsize=(20, 10))
                plt.imshow(arr)
                for k, df in enumerate(dfs):
                    plt.scatter(df['x'][i], df['y'][i], s=100, alpha=0.5, label=k)
                for df in dfs:
                    plt.plot(df['x'][:i], df['y'][:i], alpha=0.5)
                plt.legend()
                plt.savefig(f'{output_path}/frame_{i}.png')
                
                # Close the figure to free up memory
                plt.close(fig)
