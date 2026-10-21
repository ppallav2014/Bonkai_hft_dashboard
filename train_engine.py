# Author: Piyush Pallav 
# Email: ppallav2014@gmail.com
# About this code : GPU accelerated 8D neural engine.
# ========================================
import os, time, math, csv, requests, concurrent.futures
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT"]
PAGES_PER_SYMBOL, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, PATIENCE, PLOT_INTERVAL = 30, 50000, 4096, 0.005, 0.90, 1500, 50
WEIGHT_CHECKPOINT, LOG_CSV = "model_weights.npz", "training_history.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def fast_rolling_stats(arr, w=50):
    n = len(arr); mean, std = np.zeros(n), np.zeros(n)
    c1, c2 = np.cumsum(arr), np.cumsum(arr**2)
    mean[:w] = c1[:w] / np.arange(1, w+1)
    var_exp = (c2[:w] / np.arange(1, w+1)) - mean[:w]**2
    std[:w] = np.sqrt(np.maximum(var_exp, 1e-8))
    s1, s2 = c1[w:] - c1[:-w], c2[w:] - c2[:-w]
    mean[w:] = s1 / w
    var = (s2 / w) - mean[w:]**2
    std[w:] = np.sqrt(np.maximum(var, 1e-8))
    return mean, std

def fast_ema(arr, span):
    alpha = 2.0 / (span + 1.0); ema = np.zeros(len(arr)); ema[0] = arr[0]
    for i in range(1, len(arr)): ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema

def fast_rolling_sum(arr, w=50):
    res = np.zeros(len(arr)); c1 = np.cumsum(arr)
    res[:w] = c1[:w]; res[w:] = c1[w:] - c1[:-w]
    return res

def fetch_agg_trades(symbol):
    print(f"[*] Fetching {PAGES_PER_SYMBOL * 1000:,} ticks for {symbol}..."); records = []
    try:
        latest = requests.get("https://api.binance.com/api/v3/aggTrades", params={"symbol": symbol, "limit": 1}, timeout=10).json()
        if not latest: return symbol, []
        current_id = latest[0]['a'] - (PAGES_PER_SYMBOL * 1000)
        for _ in range(PAGES_PER_SYMBOL):
            chunk = requests.get("https://api.binance.com/api/v3/aggTrades", params={"symbol": symbol, "limit": 1000, "fromId": current_id}, timeout=10).json()
            if not chunk: break
            records.extend(chunk); current_id = chunk[-1]['a'] + 1
    except Exception as e: print(f"[!] Error fetching {symbol}: {e}")
    return symbol, records

def build_dataset():
    print(f"[*] Pooling RAM and pulling high-frequency ticks via multi-threading...")
    all_features, all_targets = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SYMBOLS)) as executor:
        results = executor.map(fetch_agg_trades, SYMBOLS)
        
    for sym, records in results:
        if len(records) < 1000: continue
        close, qty = np.array([float(r['p']) for r in records]), np.array([float(r['q']) for r in records])
        is_maker = np.array([r['m'] for r in records], dtype=bool)
        sell_vol, buy_vol = np.where(is_maker, qty, 0.0), np.where(~is_maker, qty, 0.0)
        m_arr, s_arr = fast_rolling_stats(close, w=50)
        
        z_score = np.clip((close - m_arr) / (s_arr + 1e-6), -4.0, 4.0)
        ret = np.zeros_like(close); ret[10:] = (close[10:] - close[:-10]) / (close[:-10] + 1e-6) * 100.0
        ret = np.clip(ret, -5.0, 5.0)
        
        roll_buy, roll_sell = fast_rolling_sum(buy_vol, w=50), fast_rolling_sum(sell_vol, w=50)
        vol_delta = np.clip((roll_buy - roll_sell) / (roll_buy + roll_sell + 1e-6), -1.0, 1.0)
        
        ema12, ema26 = fast_ema(close, 12), fast_ema(close, 26)
        macd = (ema12 - ema26) / (ema26 + 1e-6) * 100
        
        diffs = np.diff(close, prepend=close[0])
        gains, losses = np.maximum(diffs, 0), np.abs(np.minimum(diffs, 0))
        roll_gains, roll_losses = fast_rolling_stats(gains, 14)[0], fast_rolling_stats(losses, 14)[0]
        rs = roll_gains / (roll_losses + 1e-6)
        rsi_norm = ((100 - (100 / (1 + rs))) - 50) / 50.0
        
        upper, lower = m_arr + 2*s_arr, m_arr - 2*s_arr
        bb_norm = ((close - lower) / (upper - lower + 1e-6) - 0.5) * 2.0
        
        cum_vol, cum_pv = np.cumsum(qty), np.cumsum(close * qty)
        vwap = cum_pv / (cum_vol + 1e-8)
        vwap_dist = (close - vwap) / (vwap + 1e-6) * 100
        volat = s_arr / (m_arr + 1e-6) * 100
        
        target_diff = np.zeros_like(ema12); target_diff[:-50] = ema12[50:] - ema26[50:]
        target = np.where(target_diff > 0, 1.0, -1.0)
        
        features = np.column_stack([z_score, ret, vol_delta, macd, rsi_norm, bb_norm, vwap_dist, volat])[500:-50]
        t = target[500:-50]
        all_features.append(features); all_targets.append(t)
        print(f"    -> Extracted {len(features):,} 8D vectors from {sym}")
        
    X, Y = np.vstack(all_features), np.concatenate(all_targets).reshape(-1, 1)
    indices = np.arange(len(X)); np.random.seed(42); np.random.shuffle(indices)
    split = int(len(X) * 0.80)
    
    X_train, Y_train = torch.FloatTensor(X[indices[:split]]).to(DEVICE), torch.FloatTensor(Y[indices[:split]]).to(DEVICE)
    X_val, Y_val = torch.FloatTensor(X[indices[split:]]).to(DEVICE), torch.FloatTensor(Y[indices[split:]]).to(DEVICE)
    return X_train, Y_train, X_val, Y_val

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16):
        super().__init__()
        self.fc1, self.fc2, self.tanh = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, 1), nn.Tanh()
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.tanh(self.fc2(self.tanh(self.fc1(x))))

def export_weights_for_dashboard(model, path):
    W1, b1 = model.fc1.weight.detach().cpu().numpy().T, model.fc1.bias.detach().cpu().numpy().reshape(1, -1)
    W2, b2 = model.fc2.weight.detach().cpu().numpy().T, model.fc2.bias.detach().cpu().numpy().reshape(1, 1)
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2)

def run_training():
    print(f"[*] Initializing Hardware Engine: {DEVICE}")
    X_train, Y_train, X_val, Y_val = build_dataset()
    num_samples, num_batches = len(X_train), int(math.ceil(len(X_train) / BATCH_SIZE))
    print(f"\n[*] VRAM loaded: {num_samples:,} training ticks, {len(X_val):,} validation ticks.")
    print(f"[*] GPU Engine engaged. Press Ctrl+C to halt and serialize weights.\n")

    model = PyTorchMLP(input_dim=8, hidden_dim=16).to(DEVICE)
    criterion, optimizer = nn.MSELoss(), optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
    plt.style.use('dark_background'); fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor("#000000")
    for ax in (ax1, ax2): ax.set_facecolor("#000000"); ax.grid(True, color="#1e293b", linestyle="--", alpha=0.6)
        
    line_train_loss, = ax1.plot([], [], color="#00f0ff", lw=1.8, label="Train Loss (MSE)")
    line_val_loss, = ax1.plot([], [], color="#ff0055", lw=1.8, label="Validation Loss")
    ax1.set_ylabel("Loss", color="#f8fafc", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", facecolor="#000000", edgecolor="#1e293b")
    ax1.set_title("8D TICK-LEVEL PRE-TRAINING // GPU ENGINE", color="#00ff88", fontsize=12, fontweight="bold")
    
    line_val_acc, = ax2.plot([], [], color="#00ff88", lw=1.8, label="Directional Accuracy (%)")
    ax2.axhline(50.0, color="#64748b", linestyle=":", label="Noise Threshold (50%)")
    ax2.set_xlabel("Epoch", color="#f8fafc", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", color="#f8fafc", fontsize=10, fontweight="bold")
    ax2.legend(loc="lower right", facecolor="#000000", edgecolor="#1e293b")
    status_text = fig.text(0.12, 0.94, "", color="#ffb703", fontfamily="monospace", fontsize=9, fontweight="bold")
    
    plt.ion(); plt.show()
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss, epochs_without_improvement, start_time = float("inf"), 0, time.time()
    
    try:
        for epoch in range(1, EPOCHS + 1):
            perm = torch.randperm(num_samples)
            X_train_shuff, Y_train_shuff = X_train[perm], Y_train[perm]
            model.train(); batch_loss_acc = 0.0
            
            for b in range(num_batches):
                start_i, end_i = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, num_samples)
                xb, yb = X_train_shuff[start_i:end_i], Y_train_shuff[start_i:end_i]
                optimizer.zero_grad()
                y_pred = model(xb)
                loss = criterion(y_pred, yb)
                loss.backward(); optimizer.step()
                batch_loss_acc += loss.item() * (end_i - start_i)
                
            epoch_train_loss = batch_loss_acc / num_samples
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                epoch_val_loss = criterion(val_pred, Y_val).item()
                epoch_val_acc = (torch.sign(val_pred) == Y_val).float().mean().item() * 100.0
            
            train_losses.append(epoch_train_loss); val_losses.append(epoch_val_loss); val_accuracies.append(epoch_val_acc)
            if epoch_val_loss < best_val_loss:
                best_val_loss, epochs_without_improvement = epoch_val_loss, 0
                export_weights_for_dashboard(model, WEIGHT_CHECKPOINT)
            else: epochs_without_improvement += 1
                
            if epoch % PLOT_INTERVAL == 0 or epoch == 1:
                xs = list(range(1, len(train_losses) + 1))
                line_train_loss.set_data(xs, train_losses); line_val_loss.set_data(xs, val_losses); line_val_acc.set_data(xs, val_accuracies)
                ax1.set_xlim(1, max(epoch, 10)); ax1.set_ylim(0, max(max(train_losses), max(val_losses)) * 1.1)
                ax2.set_xlim(1, max(epoch, 10)); ax2.set_ylim(min(45.0, min(val_accuracies) - 2), max(65.0, max(val_accuracies) + 2))
                elapsed = time.time() - start_time
                status_text.set_text(f"EPOCH: {epoch}/{EPOCHS} | TRAIN LOSS: {epoch_train_loss:.4f} | VAL LOSS: {epoch_val_loss:.4f} | VAL ACC: {epoch_val_acc:.2f}% | ELAPSED: {elapsed:.1f}s")
                fig.canvas.draw(); fig.canvas.flush_events()
                
            if epochs_without_improvement >= PATIENCE:
                print(f"\n[!] Early stopping triggered at epoch {epoch}. Matrices saturated.")
                break

    except KeyboardInterrupt: print("\n[!] Training halted. Matrices safely serialized.")

    with open(LOG_CSV, mode="w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_Accuracy"])
        for ep, tl, vl, va in zip(range(1, len(train_losses) + 1), train_losses, val_losses, val_accuracies):
            writer.writerow([ep, f"{tl:.6f}", f"{vl:.6f}", f"{va:.2f}"])
            
    print("\n=======================================================")
    print(f"[✓] Best Validation Loss: {best_val_loss:.4f}\n[✓] 8D Checkpoint serialized: {WEIGHT_CHECKPOINT}")
    print("=======================================================")
    plt.ioff(); plt.show()

if __name__ == "__main__":
    run_training()
