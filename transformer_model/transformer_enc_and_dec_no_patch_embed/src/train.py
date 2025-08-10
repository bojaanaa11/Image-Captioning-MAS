import torch
from src.eval import evaluate
from src.save_training_info import save_training_metrics, plot_training_metrics

def train(
        encoder,
        decoder,
        criterion,
        optimizer,
        train_loader, val_loss_loader, val_metrics_loader,
        device,
        num_epochs,
        encoder_max_grad_clip_norm,
        decoder_max_grad_clip_norm,
        patience):

    best_bleu = -1
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    bleu_scores = []

    for epoch in range(num_epochs):
        # Training phase
        encoder.train()
        decoder.train()
        epoch_train_loss = 0

        print("\n Training started!\n")
        for idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            features = encoder(images)

            outputs = decoder(features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), encoder_max_grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), decoder_max_grad_clip_norm)
            optimizer.step()

            epoch_train_loss += loss.item() * images.size(0)

            if idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Calculate average training loss
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        encoder.eval()
        decoder.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            # Calculate validation loss
            print("\n Validation started!\n")
            for images, captions in val_loss_loader:
                images = images.to(device)
                captions = captions.to(device)

                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
                epoch_val_loss += loss.item() * images.size(0)

        # Calculate average validation loss
        epoch_val_loss /= len(val_loss_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Calculate validation metrics
        print("\n Evaluating started!\n")
        val_metrics = evaluate(encoder, decoder, val_metrics_loader, device)
        bleu_scores.append(val_metrics['BLEU-4'])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"BLEU-4: {val_metrics['BLEU-4']:.4f}\n")

        # Early stopping logic
        if val_metrics['BLEU-4'] > best_bleu or epoch_val_loss < best_val_loss:
            if val_metrics['BLEU-4'] > best_bleu:
                best_bleu = val_metrics['BLEU-4']

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

            no_improve = 0
            torch.save(encoder.state_dict(), '/home/obojana/bojana/src_saved/best_encoder.pth')
            torch.save(decoder.state_dict(), '/home/obojana/bojana/src_saved/best_decoder.pth')

            torch.save(optimizer.state_dict(), "/home/obojana/bojana/src_saved/optimizer.pth")
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered!")
            break

        save_training_metrics(train_losses, val_losses, bleu_scores)

    return train_losses, val_losses, bleu_scores
