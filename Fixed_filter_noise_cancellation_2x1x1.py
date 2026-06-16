import torch


class Fixed_filter_controller_2x1x1_GFANC:
    """Frame-wise fixed-filter controller used by the SF-GFANC simulation."""

    def __init__(
        self,
        frame_length: int,
        control_filters,
        num_channels: int = 2,
        filter_length: int = 1024,
    ):
        self.frame_length = frame_length
        self.num_channels = num_channels
        self.filter_length = filter_length
        self.control_filters = torch.as_tensor(control_filters, dtype=torch.float32)
        self.delay_lines = torch.zeros(num_channels, filter_length, dtype=torch.float32)
        self.current_filter = torch.zeros(num_channels, filter_length, dtype=torch.float32)

    def noise_cancellation(self, disturbance, reference):
        disturbance = torch.as_tensor(disturbance, dtype=torch.float32)
        reference = torch.as_tensor(reference, dtype=torch.float32)
        error = torch.zeros(disturbance.shape[0], dtype=torch.float32)

        frame_index = 0
        for sample_index in range(disturbance.shape[0]):
            for channel in range(self.num_channels):
                self.delay_lines[channel] = torch.roll(self.delay_lines[channel], 1, 0)
                self.delay_lines[channel, 0] = reference[sample_index, channel]

            anti_noise = torch.sum(self.current_filter * self.delay_lines)
            error[sample_index] = disturbance[sample_index] - anti_noise

            is_frame_end = (sample_index + 1) % self.frame_length == 0
            if is_frame_end and frame_index < self.control_filters.shape[0]:
                self.current_filter = self.control_filters[frame_index].clone()
                frame_index += 1

        return error
