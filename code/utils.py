import argparse
import os, tempfile


def is_valid_file(parser, arg):
    """
    Taken from
    https://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va
    and
    https://stackoverflow.com/questions/9532499/check-whether-a-path-is-valid-in-python-without-creating-a-file-at-the-paths-ta
    """
    arg = str(arg)
    if os.path.exists(arg):
        return arg

    dirname = os.path.dirname(arg) or os.getcwd()
    try:
        with tempfile.TemporaryFile(dir=dirname): pass
        return arg
    except Exception:
        parser.error("A file at the given path cannot be created: " % arg)

def print_epoch_status(epoch_num, num_batches, current_batch_num, disc_loss, gen_loss):

    prog_bar_length = 20

    progress = round(prog_bar_length * float(current_batch_num) / num_batches)

    prog_bar = ''.join(['='] * progress + ['.'] * (prog_bar_length - progress))

    out_format = "Epoch {}: [{}] (batch {} / {}) \t Discriminator Loss: {:.2f}, \t Generator Loss: {:.2f} \r"
    print(out_format.format(epoch_num, prog_bar, current_batch_num, num_batches, disc_loss, gen_loss), end="")
