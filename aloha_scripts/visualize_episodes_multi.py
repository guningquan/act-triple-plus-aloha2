import os
# Must be set before numpy is imported. On this server (THP defrag=madvise, fragmented
# memory) numpy's MADV_HUGEPAGE on large arrays makes page faults stall in kernel direct
# compaction; materialising a decoded episode this way once hung for 10+ minutes.
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np
import cv2
import h5py
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
try:
    from aloha_scripts.constants import DT
except ImportError:
    from constants import DT

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]
BASE_STATE_NAMES = ["linear_vel", "angular_vel"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        effort = root['/observations/effort'][()]
        action = root['/action'][()]
        base_action = root['/base_action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    # Compressed images are left as padded JPEG bytes; save_videos decodes them frame by frame.
    return qpos, qvel, effort, action, base_action, image_dict, compressed

import os

def parse_episode_indices(episode_input):
    """
    Parse the input episode indices, which can be a single number,
    a comma-separated list, or a range.
    Examples:
    - "2" -> [2]
    - "2,3,7-8" -> [2, 3, 7, 8]
    - "5-9" -> [5, 6, 7, 8, 9]
    """
    indices = set()  # Use a set to avoid duplicates
    parts = episode_input.split(",")  # Split the input by commas
    for part in parts:
        if "-" in part:  # Handle ranges
            start, end = map(int, part.split("-"))
            indices.update(range(start, end + 1))  # Add all numbers in the range
        else:  # Handle single numbers
            indices.add(int(part))
    return sorted(indices)  # Return a sorted list


def main(args):
    dataset_dir = args['dataset_dir']
    episode_input = args['episode_idx']  # Can be a single number, list, or range, e.g., "2,3,7-8"
    ismirror = args['ismirror']

    # Parse episode_idx to support single or multiple indices
    episode_indices = parse_episode_indices(episode_input)
    print(f"Processing episodes: {episode_indices}")

    for episode_idx in episode_indices:
        if ismirror:
            dataset_name = f'mirror_episode_{episode_idx}'
        else:
            dataset_name = f'episode_{episode_idx}'

        # Load the data
        qpos, qvel, effort, action, base_action, image_dict, compressed = load_hdf5(dataset_dir, dataset_name)
        print(f'HDF5 for {dataset_name} loaded!!')

        # Save the video
        save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'),
                    compressed=compressed)

        # Visualize joint data
        visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))

        # Other visualizations (uncomment if needed)
        print("effort",effort)
        visualize_single(effort, 'effort', plot_path=os.path.join(dataset_dir, dataset_name + '_effort.png'))
        # visualize_single(action - qpos, 'tracking_error', plot_path=os.path.join(dataset_dir, dataset_name + '_error.png'))
        visualize_base(base_action, plot_path=os.path.join(dataset_dir, dataset_name + '_base_action.png'))

        # TODO: Add timestamp visualization if needed
        # visualize_timestamp(t_list, dataset_path)

def save_videos(video, dt, video_path=None, compressed=False):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        # video: {cam_name: per-frame images (T, H, W, 3), or padded JPEG bytes if compressed}.
        # Build one frame at a time instead of materialising the whole episode (decoded
        # 4 cams x ~1600 frames peaked at ~17 GB). Worker threads decode a bounded window of
        # upcoming frames (cv2 releases the GIL) while this thread runs the encoder.
        cam_names = list(video.keys())
        n_frames = len(video[cam_names[0]])

        def get_frame(t):
            images = [cv2.imdecode(video[c][t], 1) if compressed else video[c][t] for c in cam_names]
            return np.concatenate(images, axis=1)[:, :, [2, 1, 0]]  # width dimension; swap B and R channel

        fps = int(1 / dt)
        out = None
        with ThreadPoolExecutor(max_workers=4) as ex:
            pending = deque(ex.submit(get_frame, t) for t in range(min(16, n_frames)))
            next_t = len(pending)
            while pending:
                image = pending.popleft().result()
                if next_t < n_frames:
                    pending.append(ex.submit(get_frame, next_t))
                    next_t += 1
                if out is None:
                    h, w, _ = image.shape
                    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                out.write(image)
        if out is not None:
            out.release()
        print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_single(efforts_list, label, plot_path=None, ylim=None, label_overwrite=None):
    efforts = np.array(efforts_list) # ts, dim
    num_ts, num_dim = efforts.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(efforts[:, dim_idx], label=label)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()

def visualize_base(readings, plot_path=None):
    readings = np.array(readings) # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = BASE_STATE_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'), label='smoothed_20')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'), label='smoothed_10')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'), label='smoothed_5')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # if ylim:
    #     for dim_idx in range(num_dim):
    #         ax = axs[dim_idx]
    #         ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=str, help='Episode index.', required=False)
    parser.add_argument('--ismirror', action='store_true')
    main(vars(parser.parse_args()))