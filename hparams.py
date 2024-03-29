from collections import namedtuple
from text import symbols

def convert_to_objects(**kwargs):
    return namedtuple("hparams", kwargs.keys())(*kwargs.values())


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = convert_to_objects(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],
        num_workers=6,

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        # n_symbols + 1 for stop token
        n_symbols=(len(symbols)+2),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        encoder_output_dim=512,

        # GST parameters
        gst_embedding_dim=128,
        gst_heads=4,
        gst_output_channels=[32, 32, 64, 64, 128, 128],
        gst_kernel_size=[3,3],
        gst_stride=[2,2],
        gst_padding=[1,1],

        # TPSE parameters
        tpse_rnn_size=256,

        # Duration parameters
        duration_rnn_dim=512,
        range_rnn_dim=512,
        duration_lambda=2.0,
        positional_embedding_dim=32,
        timestep_denominator=10000.0,
        lambda_duration=2.0,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dims=[512,512,512,512,128],
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=10.0,
        batch_size=128,
        mask_padding=True  # set model's padded outputs to padded values
    )

    return hparams
