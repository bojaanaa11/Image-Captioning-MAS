import torch
import torch.nn as nn
import src.parameters as params
from src.positional_encoding import PositionEncoding
from torch.nn.utils.rnn import pad_sequence

once = False

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, vocab_size, num_layers=params.transformer_decoder_num_layers, nhead=params.transfomer_decoder_nhead, dropout=params.transformer_decoder_dropout):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.positional_encoding = PositionEncoding()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        global once

        # features: (batch_size, num_pixels, 2048)
        # captions: (batch_size, caption_length)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Decoder: Features: {features.shape}\n')

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Decoder: Captions: {captions.shape}\n')

        caption_length = captions.size(1)

        # Embed the captions
        embeddings = self.embed(captions)  # (batch_size, caption_length, embed_size)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Decoder: Embeddings of captions: {embeddings.shape}\n')

        # Positional encoding of captions
        embeddings = self.positional_encoding(embeddings)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Decoder: Embeddings after positional encoding: {embeddings.shape}\n')

        # Transformer decoder expects (caption_length, batch_size, embed_size)
        embeddings = embeddings.permute(1, 0, 2)  # (caption_length, batch_size, embed_size)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Decoder: Embeddings after permute: {embeddings.shape}\n')

        # Generate masks for the decoder
        tgt_mask = self.generate_square_subsequent_mask(caption_length).to(features.device)
        tgt_padding_mask = (captions == self.vocab.stoi["<pad>"]).to(features.device)

        # Convert tgt_padding_mask to float with -inf and 0 values
        tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == 1, float('-inf'))

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Decoder: Mask: {tgt_mask.shape}\n')
                f.write(f'Decoder: Mask for pads: {tgt_padding_mask.shape}\n')

        # Pass through the transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=embeddings,                              # (caption_length, batch_size, embed_size)
            memory=features,                             # (num_pixels, batch_size, embed_size)
            tgt_mask=tgt_mask,                           # Mask to prevent attending to future tokens
            tgt_key_padding_mask=tgt_padding_mask        # Mask to ignore padding tokens
        )
        decoder_output = decoder_output.permute(1, 0, 2)  # (batch_size, caption_length, embed_size)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Decoder: Output: {decoder_output.shape}\n')

        # Predict the next word
        outputs = self.fc_out(decoder_output)  # (batch_size, caption_length, vocab_size)

        if not once:
            with open('/home/obojana/bojana/src_saved/log.txt', 'a') as f:
                f.write(f'Decoder: Outputs of fc layer: {outputs.shape}\n')

            once = True

        return outputs

    def generate(self, features, max_length=params.max_length):
        batch_size = features.size(1)  # batch_size is the second dimension
        captions = torch.full((batch_size, 1), self.vocab.stoi["<start>"], dtype=torch.long).to(features.device)

        for _ in range(max_length):
            # Embed the current captions
            embeddings = self.embed(captions)  # (batch_size, current_length, embed_size)

            # Apply positional encoding
            embeddings = self.positional_encoding(embeddings)

            # Permute embeddings to match the expected shape: [current_length, batch_size, embed_size]
            embeddings = embeddings.permute(1, 0, 2)

            # Generate masks
            current_length = captions.size(1)
            tgt_mask = self.generate_square_subsequent_mask(current_length).to(features.device)  # (current_length, current_length)
            tgt_padding_mask = (captions == self.vocab.stoi["<pad>"]).to(features.device)  # (batch_size, current_length)
            tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == 1, float('-inf'))

            # Pass through the transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=embeddings,  # (current_length, batch_size, embed_size)
                memory=features,  # (num_pixels, batch_size, embed_size)
                tgt_mask=tgt_mask,  # (current_length, current_length)
                tgt_key_padding_mask=tgt_padding_mask  # (batch_size, current_length)
            )

            # Permute decoder_output back to [batch_size, current_length, embed_size]
            decoder_output = decoder_output.permute(1, 0, 2)

            # Predict the next word
            next_word_logits = self.fc_out(decoder_output[:, -1, :])  # (batch_size, vocab_size)
            next_word = next_word_logits.argmax(1).unsqueeze(1)  # (batch_size, 1)

            # Append the predicted word to the captions
            captions = torch.cat([captions, next_word], dim=1)

            # Stop if all sequences have generated the <end> token
            if (next_word == self.vocab.stoi["<end>"]).all():
                break

        return captions

    def generate_beam(self, features, beam_width=5, max_length=params.max_length):
        batch_size = features.size(1)  # batch_size is the second dimension
        start_token = self.vocab.stoi["<start>"]
        end_token = self.vocab.stoi["<end>"]

        # List to store generated captions for each instance in the batch
        all_captions = []

        # Process each instance in the batch separately
        for i in range(batch_size):
            # Initialize beams for the current instance: (log_prob, sequence)
            beams = [(0.0, [start_token])]  # Start with a single beam containing the start token

            # Get features for the current instance
            instance_features = features[:, i:i+1, :]  # (num_pixels, 1, embed_size)

            for _ in range(max_length):
                new_beams = []

                for log_prob, sequence in beams:
                    # Stop expanding this beam if the last token is <end>
                    if sequence[-1] == end_token:
                        new_beams.append((log_prob, sequence))
                        continue

                    # Convert sequence to tensor
                    captions = torch.tensor([sequence], dtype=torch.long).to(features.device)  # (1, current_length)

                    # Embed the current captions
                    embeddings = self.embed(captions)  # (1, current_length, embed_size)

                    # Apply positional encoding
                    embeddings = self.positional_encoding(embeddings)

                    # Permute embeddings to match the expected shape: [current_length, 1, embed_size]
                    embeddings = embeddings.permute(1, 0, 2)

                    # Generate masks
                    current_length = captions.size(1)
                    tgt_mask = self.generate_square_subsequent_mask(current_length).to(features.device)  # (current_length, current_length)
                    tgt_padding_mask = (captions == self.vocab.stoi["<pad>"]).to(features.device)  # (1, current_length)
                    tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == 1, float('-inf'))

                    decoder_output = self.transformer_decoder(
                        tgt=embeddings,  # (current_length, 1, embed_size)
                        memory=instance_features,  # (num_pixels, 1, embed_size)
                        tgt_mask=tgt_mask,  # (current_length, current_length)
                        tgt_key_padding_mask=tgt_padding_mask  # (1, current_length)
                    )

                    # Permute decoder_output back to [1, current_length, embed_size]
                    decoder_output = decoder_output.permute(1, 0, 2)


                    # Predict the next word logits
                    next_word_logits = self.fc_out(decoder_output[:, -1, :])  # (1, vocab_size)
                    next_word_probs = torch.log_softmax(next_word_logits, dim=-1)  # (1, vocab_size)

                    # Get top-k candidates
                    topk_probs, topk_indices = next_word_probs.topk(beam_width, dim=-1)  # (1, beam_width)

                    # Expand the beam with top-k candidates
                    for j in range(beam_width):
                        new_log_prob = log_prob + topk_probs[0, j].item()
                        new_sequence = sequence + [topk_indices[0, j].item()]
                        new_beams.append((new_log_prob, new_sequence))

                # Keep only the top-k beams
                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

            # Store the best beam for the current instance
            best_beam = beams[0][1]  # Sequence with the highest log probability
            all_captions.append(torch.tensor(best_beam, dtype=torch.long).to(features.device))

        # Pad the captions to the same length
        all_captions = pad_sequence(all_captions, batch_first=True, padding_value=self.vocab.stoi["<pad>"])

        return all_captions

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask