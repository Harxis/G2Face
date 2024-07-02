from torchvision.transforms import transforms



def get_normalizer(channel_mean, channel_std):
    MEAN = [-mean/std for mean, std in zip(channel_mean, channel_std)]
    STD = [1/std for std in channel_std]
    normalizer = transforms.Normalize(mean=channel_mean, std=channel_std)
    denormalizer = transforms.Normalize(mean=MEAN, std=STD)
    return normalizer, denormalizer



