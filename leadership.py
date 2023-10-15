import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os

def find_peaks(displacements, threshold=1, spike_refractory_period=10):
    cutoff_frequency = 1  # Cutoff frequency of the high-pass filter (in Hz)
    order = 4  # Filter order

    # Design the high-pass Butterworth filter
    b, a = signal.butter(order, cutoff_frequency, btype='high', fs=30)
    filtered_signals = [signal.filtfilt(b, a, displacement) for displacement in displacements]

    peaks = [signal.find_peaks(filtered_signal, height=threshold, distance=spike_refractory_period)[0] for filtered_signal in filtered_signals]

    return peaks, filtered_signals

def calculate_leadership_score(tmax, leader_sequence, follower_sequence, tau=20, refractory_period=10):
    trace1 = [0]
    trace2 = [0]
    exp_kerr = np.exp(-1/tau)
    w = [0]
    spike_times = [-1000]
    for i in range(1, tmax):
        trace1.append(trace1[-1])
        trace2.append(trace2[-1])
        if i in follower_sequence and i - spike_times[-1] > refractory_period:
            trace1[i]+=1
            w.append(w[-1] - trace2[i])
            spike_times.append(i)
        if i in leader_sequence and i - spike_times[-1] > refractory_period:
            trace2[i]+=1
            w.append(w[-1] + trace1[i])
            spike_times.append(i)
        trace1[i]*=exp_kerr
        trace2[i]*=exp_kerr

    spike_times = spike_times[1:]
    w = w[1:]
    
    return w, spike_times, trace1, trace2

def stretch_leadership_score(w, spike_times, tmax):
    w_stretched = np.zeros(tmax)
    for idx, i in enumerate(spike_times):
        w_stretched[i] = w[idx]
    
    x = np.where(w_stretched != 0)[0]  # indices where w is not zero
    y = w_stretched[x]  # non-zero elements of w
    xi = np.arange(len(w_stretched))  # all indices
    w_stretched = np.interp(xi, x, y)  # interpolate

    return w_stretched

def draw_frame(ax, frame_number):
    ax.vlines(frame_number, -2, 2, color='black')

def lidership_plot(dfs, leader_id, follower_id, video_array, threshold=1, refractory_period=10, tau=20):
    
    displacements = [np.abs(np.diff(np.sqrt(df['x']**2 + df['y']**2))) for df in dfs]
    tmax = len(displacements[0])
    peaks, _ = find_peaks(displacements, threshold=threshold)
    w, spike_times, _, _ = calculate_leadership_score(tmax, leader_sequence=peaks[leader_id], follower_sequence=peaks[follower_id], tau=tau, refractory_period=refractory_period)
    w_stretched = stretch_leadership_score(w, spike_times, tmax)

    if not os.path.exists(f'/home/roman/Documents/code/SmartNets_Seminars/hackathon/0066/pics_{leader_id}_{follower_id}'):
        os.mkdir(f'/home/roman/Documents/code/SmartNets_Seminars/hackathon/0066/pics_{leader_id}_{follower_id}')

    for i, frame in enumerate(video_array[::30]):
        fig = plt.figure(figsize=(20, 20))
        gs = gridspec.GridSpec(3, 1)  # 3 rows, 1 column

        ax0 = plt.subplot(gs[0:2, :])  # First two rows for the first plot
        ax0.imshow(frame)
        for k, df in enumerate(dfs):
            alpha = 0.7 if k in [leader_id, follower_id] else 0.2
            ax0.scatter(df['x'][i*30], df['y'][i*30], s=100, alpha=alpha, label=k)
        for k, df in enumerate(dfs):
            alpha = 0.7 if k in [leader_id, follower_id] else 0.2
            ax0.plot(df['x'][:i*30], df['y'][:i*30], alpha=alpha)
        ax0.legend()

        ax1 = plt.subplot(gs[2, :])  # Last row for the second plot
        ax1.plot(w_stretched, color='black')
        ax1.fill_between(range(len(w_stretched)), w_stretched, where=(w_stretched >= 0), color=f'C{leader_id}', interpolate=True)  # green for above zero
        ax1.fill_between(range(len(w_stretched)), w_stretched, where=(w_stretched <= 0), color=f'C{follower_id}', interpolate=True)  # red for below zero
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Leadership score')
        draw_frame(ax1, i*30)

        
        plt.savefig(f'/home/roman/Documents/code/SmartNets_Seminars/hackathon/0066/pics_{leader_id}_{follower_id}/frame_{i}.png')

        # Close the figure to free up memory
        plt.close(fig)

def get_dfs(group):
    dfs = []
    for i in range(10):
        data_dict = {}
        for key, dset in group.items():
            # if dset.shape and dset.shape[0] == 10:
            if i == 1 : print(key, dset.shape)
            # if key in ['x', 'y']:
            if dset.shape and len(dset.shape) == 2:
                
                data_dict[key] = dset[()][i]
        dfs.append(pd.DataFrame(data_dict))
    return dfs

def get_video():
    # Initialize an empty list to store the frames
    import av
    container = av.open('/home/roman/Documents/code/SmartNets_Seminars/hackathon/0066/0066_000-calib.mov')

    video_frames = []

    packets = list(container.demux())

    for packet in packets:
        for frame in packet.decode():
            # if frame.type == 'video':
            img = frame.to_image()  # PIL/Pillow image
            arr = np.asarray(img)
            # Append the frame to the list
            video_frames.append(arr)

    # Convert the list of frames into a numpy array
    video_array = np.array(video_frames)
    return video_array



if __name__ == '__main__':
    import h5py
    f = h5py.File('/home/roman/Documents/code/SmartNets_Seminars/hackathon/0066/0066_000_fov.h5', 'r')
    group = f['fields']
    dfs = get_dfs(group)
    video_array = get_video()
    leadership_plot(dfs, leader_id=0, follower_id=1, video_array=video_array, threshold=1, refractory_period=10, tau=20)