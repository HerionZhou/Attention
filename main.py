import pickle
import time
import argparse
import os

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from model import EncoderCNN, AttnDecoderRNN

from data_loader import get_loader
from nltk.translate.bleu_score import corpus_bleu
from utils import *
from caption import caption_image_beam_search
from build_vocab import Vocabulary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='coco_5_cap_per_img_5_min_word_freq')
parser.add_argument('--model_path', type=str, default='./model' , help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')
parser.add_argument('--image_dir', type=str, default='./data/resized', help='directory for resized images')
parser.add_argument('--caption_path', type=str, default='./data/annotations/karpathy_split_train.json', help='path for train annotation json file')
parser.add_argument('--caption_path_val', type=str, default='./data/annotations/karpathy_split_val.json', help='path for val annotation json file')
parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

# Model parameters
parser.add_argument('--embed_dim', type=int, default=512, help='dimension of word embedding vectors')
parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers')
parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder rnn')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--epochs_since_improvement', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--encoder_lr', type=float, default=1e-4)
parser.add_argument('--decoder_lr', type=float, default=4e-4)
parser.add_argument('--checkpoint', type=str, default='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' , help='path for checkpoints')
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--best_bleu4', type=float, default=0.)
parser.add_argument('--fine_tune_encoder', type=bool, default='False' , help='fine-tune encoder')
parser.add_argument('--beam_size', type=int, default=3)

args = parser.parse_args()
print(args)


def main(args):
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, beam_size
    beam_size = args.beam_size
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    word_map = vocab

    best_bleu4 = 0

    if not os.path.exists(args.checkpoint):
        decoder = AttnDecoderRNN(attention_dim=args.attention_dim,
                                 embed_dim=args.embed_dim,
                                 decoder_dim=args.decoder_dim,
                                 vocab_size=len(vocab),
                                 dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),lr=args.decoder_lr)
        encoder = EncoderCNN()
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None
    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Build data loader
    train_loader = get_loader(args.image_dir, args.caption_path, vocab,
                              transform, args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(args.image_dir, args.caption_path_val, vocab,
                            transform, batch_size=1,
                            shuffle=True, num_workers=args.num_workers)

    for epoch in range(args.start_epoch, args.epochs):
        if args.epochs_since_improvement == 20:
            break
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            args.epochs_since_improvement +=1
            print("\nEpoch since last improvement: %d\n" %(args.epochs_since_improvement,))
        else:
            args.epochs_since_improvement = 0

        save_checkpoint(args.data_name, epoch, args.epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                        recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        imgs = encoder(imgs)

        # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

        targets = caps_sorted[:, 1:]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        loss = criterion(scores, targets)
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args.log_step == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (img, caps, caplen) in enumerate(val_loader):

        seq, alphas = caption_image_beam_search(encoder, decoder, img, word_map, beam_size)

        if i % (args.log_step / 10) == 0:
            print('Validation: [{0}/{1}]\t'.format(i, len(val_loader)))

        # References
        # caps = caps[sort_ind]  # because images were sorted in the decoder
        img_caps = caps[0].tolist()

        # img_captions = list(
        #     map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
        #         img_caps))  # remove <start> and pads
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map('<start>'), word_map('<end>'), word_map('<pad>')}],
                [img_caps]))  # remove <start> and pads

        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map('<start>'), word_map('<end>'), word_map('<pad>')}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses, emulate_multibleu=True)

    print('\n * BLEU-4 - {bleu}\n'.format(bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main(args)
