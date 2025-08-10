import torch

from src.eval import evaluate

def test(encoder, decoder, test_loader, device):
    print('\n Testing started!')
    # Load best model checkpoint if using early stopping
    encoder.load_state_dict(torch.load('/home/obojana/bojana/src_saved/best_encoder.pth'))
    decoder.load_state_dict(torch.load('/home/obojana/bojana/src_saved/best_decoder.pth'))

    # Final evaluation
    final_metrics = evaluate(encoder, decoder, test_loader, device)

    with open('test.log', 'w') as f:
        f.write(str(final_metrics['BLEU-4']))
        f.write('\n')

    print("\nFinal Test Metrics:")
    print(f"BLEU-4: {final_metrics['BLEU-4']:.4f}")