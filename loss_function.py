from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self, lambda_duration=2.0):
        super(Tacotron2Loss, self).__init__()

        self.lambda_duration = lambda_duration

    def forward(self, model_output, targets):
        mel_target, duration_target = targets[0], targets[1]
        mel_target.requires_grad = False
        duration_target.requires_grad = False

        mel_out, mel_out_postnet, duration_out, alignment = model_output
        #mel_out, mel_out_postnet, duration_out, alignment, gst_out, gst_target = model_output

        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target) + \
            nn.L1Loss()(mel_out, mel_target) + \
            nn.L1Loss()(mel_out_postnet, mel_target)

        dur_loss = nn.MSELoss()(duration_out, duration_target)

#        tpse_loss = nn.MSELoss()(gst_out, gst_target)

        losses = mel_loss, dur_loss
        #losses = mel_loss, tpse_loss, dur_loss

        return mel_loss + self.lambda_duration * dur_loss, losses
        #return mel_loss + tpse_loss + self.lambda_duration * dur_loss, losses
