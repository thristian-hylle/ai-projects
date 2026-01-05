import torch
import torch.nn as nn
import torch.optim as optim

def train_validate_rnn(enc, dec, train_loader, val_loader, vocab_size,
                            epochs=12, lr=1e-3, max_len=128, clip=1.0,
                            save_path="best_part2_rnn.pth"):
    enc, dec = enc.to(device), dec.to(device)
    optimizer = optim.Adam(list(enc.parameters()) +list(dec.parameters()),lr=lr)

    # IMPORTANT: targets keep PAD_VALUE so ignore_index works
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_VALUE)

    best_val_LA =-1.0
    train_losses = []
    val_losses = []
    val_LAs = []

    for ep in range(epochs):
        # train
        enc.train(); dec.train()
        loss_sum = 0.0
        token_correct, token_total = 0, 0

        for x, y in train_loader:
            x, y= x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)

            _, (h, c) = enc(x)

            # decoder input: shift right, BUT replace PAD_VALUE with PAD_IDX
            dec_in = y[:,:-1].clone()
            dec_in[dec_in == PAD_VALUE] = PAD_IDX

            # target: shift left keep PAD_VALUE here so loss can ignore those positions
            target = y[:,1:].clone()

            # forward through decoder to get logits for each timestep
            # flatten batch and time dims so cross entropy sees N, V vs N
            # backprop through both encoder and decoder
            out, _= dec(dec_in, (h, c))  # (B, T, V)
            loss = criterion(out.reshape(-1, vocab_size),target.reshape(-1))
            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) +list(dec.parameters()),clip)

            optimizer.step()
            loss_sum +=loss.item()

            # debug token accuracy
            preds = out.argmax(-1)
            mask = target!= PAD_VALUE
            token_correct +=(preds[mask] == target[mask]).sum().item()
            token_total += mask.sum().item()

        # average train loss per batch
        train_loss =loss_sum / max(len(train_loader), 1)
        train_tok_acc = token_correct /max(token_total, 1)
        train_losses.append(train_loss)

        # vaildate loss + LA
        enc.eval();dec.eval()
        val_loss_sum= 0.0
        val_token_correct,val_token_total = 0, 0

        with torch.no_grad():
            for x,y in val_loader:
                x, y= x.to(device),y.to(device)
                _, (h, c) =enc(x)

                #training for validation loss
                dec_in = y[:,:-1].clone()
                dec_in[dec_in == PAD_VALUE] =PAD_IDX
                target =y[:, 1:].clone()
                out,_ = dec(dec_in, (h, c))

                # compute validation
                loss =criterion(out.reshape(-1,vocab_size),target.reshape(-1))
                val_loss_sum += loss.item()

                # debug token accuracy
                preds = out.argmax(-1)
                mask =target !=PAD_VALUE
                val_token_correct += (preds[mask] ==target[mask]).sum().item()
                val_token_total+= mask.sum().item()

        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_tok_acc = val_token_correct / max(val_token_total, 1)

        #levenshtein based sequence similarity using greedy decoding
        val_LA =levenshtein_accuracy(enc, dec, val_loader, max_len=max_len)
        val_losses.append(val_loss)
        val_LAs.append(val_LA)

        print(f"Epoch {ep+1:02d}: "f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "f"train_tok_acc={100*train_tok_acc:.2f}% | val_tok_acc={100*val_tok_acc:.2f}% | "f"VAL_LA={val_LA:.2f}%")

        # save best
        if val_LA > best_val_LA:
            best_val_LA = val_LA
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()}, save_path)
            print(f" saved {save_path} (best VAL_LA={best_val_LA:.2f}%)")

    return train_losses, val_losses, val_LAs
