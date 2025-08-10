
import torch
from pycocoevalcap.bleu.bleu import Bleu
import src.parameters as params

def evaluate(encoder, decoder, loader, device):
    encoder.eval()
    decoder.eval()
    results = []
    references = {}  # Will store {img_id: [str, str...]}

    with torch.no_grad():
        for images, captions_list, img_ids in loader:
            images = images.to(device)
            features = encoder(images)
            generated = decoder.generate_beam(features, params.beam_size, params.max_length)

            # Process captions
            for i in range(generated.size(0)):
                img_id = img_ids[i]
                caption = generated[i].cpu().tolist()
                words = []
                for idx in caption:
                    word = decoder.vocab.itos.get(idx, "<unk>")
                    if word == "<end>": break
                    if word not in ["<start>", "<pad>"]:
                        words.append(word)
                results.append({"image_id": img_id, "caption": " ".join(words)})
                references[img_id] = captions_list[i]  # Store as list of strings

    # Convert to COCO format
    res = {item["image_id"]: [item["caption"]] for item in results}  # List of strings
    gts = references  # {img_id: [str, str...]}

    # Compute metrics
    bleu_scores = Bleu(4).compute_score(gts, res)

    return {
        'BLEU-4': bleu_scores[0][3],  # BLEU-4 score
    }