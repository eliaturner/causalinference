import numpy as np
import matplotlib.pyplot as plt


class RNNTaskDataset:
    def __init__(self, n_trials, time, n_channels):
        self.n_trials = n_trials
        self.time = time
        self.n_channels = n_channels

    def ready_set_go(self):
        x = np.zeros((self.n_trials, self.time, self.n_channels))
        y = np.full((self.n_trials, self.time, 1), np.nan)  # Output with 1 channel

        for trial in range(self.n_trials):
            pulse_duration = np.random.randint(3, 6)
            ready_time = np.random.randint(10, self.time // 4)
            set_time = ready_time + np.random.randint(10, self.time // 4)
            go_time = set_time + (set_time - ready_time)

            if go_time + pulse_duration < self.time:
                x[trial, ready_time:ready_time + pulse_duration, 0] = 1  # Ready pulse on channel 0
                x[trial, set_time:set_time + pulse_duration, 1] = 1  # Set pulse on channel 1
                y[trial, go_time:go_time + pulse_duration, 0] = 1  # Go response on output
                y[trial, :go_time, 0] = 0
                # y[trial, go_time + pulse_duration:, 0] = 1

        return x, y

    def delay_discrimination(self):
        x = np.zeros((self.n_trials, self.time, self.n_channels))
        y = np.full((self.n_trials, self.time, 1), np.nan)  # Output with 1 channel

        for trial in range(self.n_trials):
            pulse_duration = 5
            # Randomly choose start times for two pulses
            start_time1 = np.random.randint(10, self.time // 4)
            start_time2 = start_time1 + np.random.randint(10, self.time // 4)

            # Generate two pulse amplitudes
            pulse_amplitude1 = np.random.uniform(0.2, 1.0)  # Amplitude from a range
            pulse_amplitude2 = np.random.uniform(0.2, 1.0)  # Amplitude from a range

            # Apply pulses to input channels
            x[trial, start_time1:start_time1 + pulse_duration, 0] = pulse_amplitude1
            x[trial, start_time2:start_time2 + pulse_duration, 1] = pulse_amplitude2

            # Compare the two pulse amplitudes and set output
            if pulse_amplitude2 > pulse_amplitude1:
                y[trial, start_time2 + pulse_duration + 5: start_time2 + pulse_duration + 10, 0] = 1  # Second pulse larger
            else:
                y[trial, start_time2 + pulse_duration + 5: start_time2 + pulse_duration + 10, 0] = -1  # First pulse larger

            y[trial, :start_time2 + pulse_duration, 0] = 0

        return x, y

    def flip_flop(self):
        x = np.zeros((self.n_trials, self.time, self.n_channels))
        y = np.full((self.n_trials, self.time, self.n_channels), np.nan)  # Output with 2 channels

        for trial in range(self.n_trials):
            state = [np.nan, np.nan]  # Initial state for both channels
            for t in range(self.time):
                if np.random.rand() > 0.95:  # Randomly trigger pulse on one channel
                    pulse_channel = np.random.choice([0, 1])
                    pulse_value = 1 if np.random.rand() > 0.5 else -1
                    x[trial, t:t + 5, pulse_channel] = pulse_value
                    state[pulse_channel] = pulse_value
                y[trial, t, 0] = state[0]  # Output maintains last pulse value on channel 0
                y[trial, t, 1] = state[1]  # Output maintains last pulse value on channel 1

        return x, y

    def evidence_accumulation(self):
        x = np.zeros((self.n_trials, self.time, self.n_channels))
        y = np.full((self.n_trials, self.time, 1), np.nan)  # Output with 1 channel

        for trial in range(self.n_trials):
            pulses_ch1 = np.random.randint(3, 10)  # Random number of pulses for channel 1
            pulses_ch2 = np.random.randint(3, 10)  # Random number of pulses for channel 2
            pulse_duration = 5

            pulse_times_ch1 = np.random.choice(np.arange(self.time - pulse_duration), pulses_ch1, replace=False)
            pulse_times_ch2 = np.random.choice(np.arange(self.time - pulse_duration), pulses_ch2, replace=False)

            for t in pulse_times_ch1:
                x[trial, t:t + pulse_duration, 0] = 1  # Pulses on channel 1
            for t in pulse_times_ch2:
                x[trial, t:t + pulse_duration, 1] = 1  # Pulses on channel 2

            if pulses_ch1 > pulses_ch2:
                y[trial, -pulse_duration:, 0] = 1  # Positive response
            else:
                y[trial, -pulse_duration:, 0] = -1  # Negative response

        return x, y

    def integrator(self):
        n_channels = 1
        x = np.zeros((self.n_trials, self.time, 1))
        y = np.zeros((self.n_trials, self.time, 1))

        for n in range(self.n_trials):
            x[n, 1:] = np.random.randint(2, size=sequence_length - 1) * 2 - 1
            y[n, 1:] = np.cumsum(x[n, 1:])

        return x, y

    def plot_task(self, x, y, title):
        time_steps = np.arange(x.shape[1])

        plt.figure(figsize=(12, 6))

        # Plot inputs
        plt.subplot(2, 1, 1)
        for channel in range(x.shape[2]):
            plt.plot(time_steps, x[0, :, channel], label=f'Input Channel {channel + 1}')
        plt.title(f"Input for {title}")
        plt.ylabel('Amplitude')
        plt.legend()

        # Plot outputs
        plt.subplot(2, 1, 2)
        for channel in range(y.shape[2]):
            plt.plot(time_steps, y[0, :, channel], label=f'Output Channel {channel + 1}')
        plt.title(f"Output for {title}")
        plt.ylabel('Output')
        plt.xlabel('Time Steps')
        plt.legend()

        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    # Example usage:
    task_dataset = RNNTaskDataset(n_trials=200, time=150, n_channels=2)

    ready_set_go_x, ready_set_go_y = task_dataset.ready_set_go()
    delay_discrimination_x, delay_discrimination_y = task_dataset.delay_discrimination()
    flip_flop_x, flip_flop_y = task_dataset.flip_flop()
    evidence_accumulation_x, evidence_accumulation_y = task_dataset.evidence_accumulation()

    # Plot each task to verify
    task_dataset.plot_task(ready_set_go_x, ready_set_go_y, "Ready Set Go")
    task_dataset.plot_task(delay_discrimination_x, delay_discrimination_y, "Delay Discrimination")
    task_dataset.plot_task(flip_flop_x, flip_flop_y, "Flip Flop")
    task_dataset.plot_task(evidence_accumulation_x, evidence_accumulation_y, "Evidence Accumulation")