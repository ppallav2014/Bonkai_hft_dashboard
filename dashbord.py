# Author: Piyush Pallav 
# Contact: ppallav2014@gmail.com
# Project: BONKAI Neural HFT Terminal
# About this code : Realtime MLP tick trading dashboard.
import sys, time, math, json, random, asyncio, threading, csv, os
from collections import deque
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QPushButton, QFrame, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QLinearGradient, QPainterPath
import websockets

USD_TO_INR = 87.50
BG_BLACK, BG_CARD, BORDER_DIM = "#000000", "#05070B", "#161D2B"
TEXT_WHITE, TEXT_MUTED = "#FFFFFF", "#64748B"
NEON_CYAN, NEON_GREEN, NEON_AMBER = "#00F0FF", "#00FF88", "#FFB703"
NEON_MAG, NEON_RED, NEON_LILAC = "#D946EF", "#FF0055", "#8B5CF6"

def create_smooth_path(pts):
    path = QPainterPath()
    if not pts: return path
    path.moveTo(pts[0])
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        ctrl1, ctrl2 = QPointF((p1.x() + p2.x()) / 2, p1.y()), QPointF((p1.x() + p2.x()) / 2, p2.y())
        path.cubicTo(ctrl1, ctrl2, p2)
    return path

class HFTLogger:
    def __init__(self, filepath="hft_diagnostic_log.csv"):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                csv.writer(f).writerow(["Timestamp", "Price", "Z_Score", "Net_Volume_Delta", "MACD", "RSI", "VWAP_Dist", "Neural_Loss", "Action", "Realized_PnL"])

    def log_event(self, price, z, vi, macd, rsi, vwap, loss, action, pnl):
        with open(self.filepath, mode='a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], price, round(z, 4), round(vi, 3), round(macd, 4), round(rsi, 2), round(vwap, 4), round(loss, 4), action, round(pnl, 3)])

class MarketWorker(QObject):
    raw_tick = pyqtSignal(float, float, str, bool)
    def __init__(self):
        super().__init__()
        self.running = True
        
    def start(self): 
        threading.Thread(target=self._loop, daemon=True).start()
        
    def _loop(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(self._stream())
        
    async def _stream(self):
        uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
        while self.running:
            try:
                async with websockets.connect(uri) as ws:
                    while self.running:
                        data = json.loads(await ws.recv())
                        self.raw_tick.emit(float(data['p']), float(data['q']), time.strftime('%H:%M:%S', time.localtime(data['T']/1000.0)), data['m'])
            except Exception:
                await asyncio.sleep(1)

class FastMLP:
    def __init__(self, input_size=8, hidden_size=16, weight_path="model_weights.npz"):
        self.lr, self.weight_path, self.input_size = 0.005, weight_path, input_size
        if os.path.exists(self.weight_path):
            try:
                data = np.load(self.weight_path)
                if data['W1'].shape[0] == input_size:
                    self.W1, self.b1, self.W2, self.b2 = data['W1'], data['b1'], data['W2'], data['b2']
                else:
                    self.init_weights(input_size, hidden_size)
            except Exception:
                self.init_weights(input_size, hidden_size)
        else:
            self.init_weights(input_size, hidden_size)
            
        self.mW1, self.mb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.mW2, self.mb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.loss, self.steps = 0.0, 0
        self.wh = [deque(maxlen=80) for _ in range(input_size)]
        self.w = np.mean(np.abs(self.W1), axis=1)

    def init_weights(self, i, h):
        self.W1 = np.random.randn(i, h) * np.sqrt(2.0 / (i + h))
        self.b1 = np.zeros((1, h))
        self.W2 = np.random.randn(h, 1) * np.sqrt(2.0 / (h + 1))
        self.b2 = np.zeros((1, 1))
        
    def predict(self, x):
        self.Z1 = np.dot(x.reshape(1, -1), self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.tanh(self.Z2)
        return float(self.A2[0, 0])
        
    def update(self, x, target):
        p = self.predict(x)
        err = target - p
        x_mat = x.reshape(1, -1)
        
        dZ2 = np.array([[-(target - p) * (1.0 - p**2)]])
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (1.0 - self.A1**2)
        dW1 = np.dot(x_mat.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        self.mW2 = 0.9 * self.mW2 + 0.1 * dW2; self.W2 -= self.lr * self.mW2
        self.mb2 = 0.9 * self.mb2 + 0.1 * db2; self.b2 -= self.lr * self.mb2
        self.mW1 = 0.9 * self.mW1 + 0.1 * dW1; self.W1 -= self.lr * self.mW1
        self.mb1 = 0.9 * self.mb1 + 0.1 * db1; self.b1 -= self.lr * self.mb1
        
        self.loss = self.loss * 0.95 + (err**2) * 0.05
        self.steps += 1
        self.w = np.mean(np.abs(self.W1), axis=1) 
        for i in range(self.input_size): self.wh[i].append(self.w[i])
        return self.loss

    def save_weights(self):
        np.savez(self.weight_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

class MarketCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.candles, self.vol_hist = [], []
        self.cur_candle = None
        self.dp = self.tp = self.vb = self.vs = self.tot_b = self.tot_s = 0.0
        self.last_c = self.last_p = time.time()
        self.target_paths = self.display_paths = None
        self.n_paths, self.n_steps = 24, 18
        
    def push(self, p, q, m):
        self.dp = p if self.tp == 0 else self.dp
        self.tp = p
        if m: 
            self.vs += q; self.tot_s += q
        else: 
            self.vb += q; self.tot_b += q
            
    def step(self):
        if self.tp == 0: return
        self.dp += (self.tp - self.dp) * 0.18
        now = time.time()
        if not self.cur_candle:
            self.cur_candle = {'o': self.dp, 'h': self.dp, 'l': self.dp, 'c': self.dp, 't': now}
            self.last_c = now
        else:
            self.cur_candle['h'] = max(self.cur_candle['h'], self.dp)
            self.cur_candle['l'] = min(self.cur_candle['l'], self.dp)
            self.cur_candle['c'] = self.dp
            self.cur_candle['t'] = now
            
        if now - self.last_c >= 1.5:
            self.candles.append(self.cur_candle)
            tv = self.vb + self.vs
            self.vol_hist.append((tv, self.vb / (tv or 1.0)))
            if len(self.candles) > 35:
                self.candles.pop(0)
                self.vol_hist.pop(0)
            self.cur_candle = {'o': self.dp, 'h': self.dp, 'l': self.dp, 'c': self.dp, 't': now}
            self.vb = self.vs = 0.0
            self.last_c = now
            
        if now - self.last_p > 0.4:
            self._paths()
            self.last_p = now
            
        if self.target_paths is not None:
            if self.display_paths is None:
                self.display_paths = np.copy(self.target_paths)
            else:
                self.display_paths += (self.target_paths - self.display_paths) * 0.12
                self.display_paths[:, 0] = self.dp
        self.update()
        
    def _paths(self):
        if len(self.candles) < 3 or self.dp == 0: return
        v = float(np.std(np.diff(np.log([c['c'] for c in self.candles])))) if len(self.candles) > 3 else 0.0003
        arr = np.zeros((self.n_paths, self.n_steps))
        for i in range(self.n_paths):
            cp = self.dp
            arr[i, 0] = cp
            for s in range(1, self.n_steps):
                cp *= math.exp(-0.5 * max(v, 0.0001)**2 + random.gauss(0, max(v, 0.0001)))
                arr[i, s] = cp
        self.target_paths = arr
        
    def paintEvent(self, e):
        pr = QPainter(self)
        pr.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pr.fillRect(0, 0, w, h, QColor(BG_BLACK))
        if not self.candles: return
        
        ch, time_axis_h, vh = int(h * 0.72), int(h * 0.05), int(h * 0.18)
        cw, cw_vol = w - 65, int((w - 65) * 0.72)
        vy = ch + time_axis_h + 4
        all_c = self.candles + [self.cur_candle] if self.cur_candle else self.candles
        mip, map_p = min(c['l'] for c in all_c), max(c['h'] for c in all_c)
        if self.display_paths is not None:
            mip = min(mip, float(np.min(self.display_paths)))
            map_p = max(map_p, float(np.max(self.display_paths)))
        rng = max(map_p - mip, self.dp * 0.0005)
        mip -= rng * 0.05
        rng *= 1.1
        
        def ty(val): return ch * (1.0 - (val - mip) / rng)
        sw = cw_vol / 36.0
        bw = max(sw * 0.7, 2)
        pr.setPen(QPen(QColor(BORDER_DIM), 1, Qt.DashLine))
        
        for i in range(5):
            y = ch * (i / 4.0)
            pr.drawLine(0, int(y), cw, int(y))
            pr.setPen(QColor(TEXT_MUTED))
            pr.setFont(QFont("Monospace", 7))
            pr.drawText(cw + 5, int(y) + 3, f"{mip + rng * (1 - i / 4):.1f}")
            pr.setPen(QPen(QColor(BORDER_DIM), 1, Qt.DashLine))
            
        pr.setFont(QFont("Monospace", 7))
        for i, c in enumerate(all_c):
            cx = i * sw + sw * 0.5
            col = QColor(NEON_GREEN) if c['c'] >= c['o'] else QColor(NEON_RED)
            pr.setPen(QPen(col, 1.5))
            pr.drawLine(int(cx), int(ty(c['h'])), int(cx), int(ty(c['l'])))
            pr.fillRect(int(cx - bw / 2), int(min(ty(c['o']), ty(c['c']))), int(bw), max(int(abs(ty(c['c']) - ty(c['o']))), 2), col)
            if i % 6 == 0:
                pr.setPen(QColor(TEXT_MUTED))
                pr.drawText(int(cx - 20), ch + 14, time.strftime('%H:%M:%S', time.localtime(c['t'])))

        lx, ly = (len(all_c) - 1) * sw + sw * 0.5, ty(self.dp)
        if self.display_paths is not None:
            p10, p90 = np.percentile(self.display_paths, 10, axis=0), np.percentile(self.display_paths, 90, axis=0)
            top_pts = [QPointF(lx + s * sw, ty(p90[s])) for s in range(self.n_steps)]
            bot_pts = [QPointF(lx + s * sw, ty(p10[s])) for s in reversed(range(self.n_steps))]
            poly_path = QPainterPath()
            poly_path.moveTo(top_pts[0])
            for s in range(1, self.n_steps):
                poly_path.cubicTo(QPointF((top_pts[s-1].x() + top_pts[s].x()) / 2, top_pts[s-1].y()), QPointF((top_pts[s-1].x() + top_pts[s].x()) / 2, top_pts[s].y()), top_pts[s])
            for s in range(self.n_steps):
                p1, p2 = top_pts[-1] if s == 0 else bot_pts[s-1], bot_pts[s]
                poly_path.cubicTo(QPointF((p1.x() + p2.x()) / 2, p1.y()), QPointF((p1.x() + p2.x()) / 2, p2.y()), p2)
            
            grad = QLinearGradient(lx, 0, lx + self.n_steps * sw, 0)
            grad.setColorAt(0, QColor(0, 240, 255, 90)); grad.setColorAt(1, QColor(0, 240, 255, 0))
            pr.setPen(Qt.NoPen); pr.setBrush(grad); pr.drawPath(poly_path)
            pr.setPen(QPen(QColor(0, 240, 255, 45), 1.2))
            for i in range(self.n_paths):
                pts = [QPointF(lx + s * sw, ty(self.display_paths[i, s])) for s in range(self.n_steps)]
                pr.drawPath(create_smooth_path(pts))
                
        pr.setPen(QPen(QColor(NEON_AMBER), 1, Qt.DashLine))
        pr.drawLine(0, int(ly), cw, int(ly))
        pr.fillRect(cw + 2, int(ly - 8), 58, 16, QColor(NEON_AMBER))
        pr.setPen(QColor(BG_BLACK))
        pr.setFont(QFont("Monospace", 7, QFont.Bold))
        pr.drawText(cw + 4, int(ly + 4), f"{self.dp:.1f}")
        
        mv = max([v[0] for v in self.vol_hist] + [self.vb + self.vs, 0.001])
        pr.fillRect(0, vy, cw_vol, vh, QColor(BG_BLACK))
        for i, (tv, br) in enumerate(self.vol_hist):
            vx, bh = i * sw + sw * 0.5, (tv / mv) * (vh - 2)
            col = QColor(NEON_GREEN) if br >= 0.5 else QColor(NEON_RED)
            pr.fillRect(int(vx - bw / 2), int(vy + vh - bh), int(bw), int(bh), col)
            
        pr.fillRect(cw_vol + 8, vy, w - cw_vol - 8, vh, QColor(BG_BLACK))
        pr.setPen(QPen(QColor(BORDER_DIM), 1))
        pr.drawRect(cw_vol + 8, vy, w - cw_vol - 8, vh)
        pr.setFont(QFont("Monospace", 8, QFont.Bold))
        pr.setPen(QColor(NEON_GREEN)); pr.drawText(cw_vol + 14, vy + 16, f"BUY VOL:  {self.tot_b:.3f}")
        pr.setPen(QColor(NEON_RED));   pr.drawText(cw_vol + 14, vy + 34, f"SELL VOL: {self.tot_s:.3f}")
        diff = self.tot_b - self.tot_s
        pr.setPen(QColor(NEON_CYAN) if diff >= 0 else QColor(NEON_AMBER))
        pr.drawText(cw_vol + 14, vy + 52, f"DELTA:    {diff:+.3f}")

class SpectralCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.d = deque(maxlen=40)
    def push(self, v): 
        self.d.append(v)
        self.update()
    def paintEvent(self, e):
        pr = QPainter(self); pr.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pr.fillRect(0, 0, w, h, QColor(BG_BLACK))
        if len(self.d) < 2: return
        my, sw = h / 2.0, w / max(1, len(self.d) - 1)
        pr.setPen(QPen(QColor(BORDER_DIM), 1)); pr.drawLine(0, int(my), w, int(my))
        pts = [QPointF(i * sw, my - (v / 1.0) * (my - 5)) for i, v in enumerate(self.d)]
        path = create_smooth_path(pts)
        fill = QPainterPath(path); fill.lineTo(w, my); fill.lineTo(0, my); fill.closeSubpath()
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(0, 240, 255, 70)); grad.setColorAt(0.5, QColor(0, 240, 255, 0)); grad.setColorAt(1.0, QColor(217, 70, 239, 70))
        pr.setBrush(grad); pr.setPen(Qt.NoPen); pr.drawPath(fill)
        pr.setPen(QPen(QColor(NEON_MAG), 2)); pr.drawPath(path)

class ZScoreCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.d = deque(maxlen=60)
    def push(self, v): 
        self.d.append(v)
        self.update()
    def paintEvent(self, e):
        pr = QPainter(self); pr.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pr.fillRect(0, 0, w, h, QColor(BG_BLACK))
        if len(self.d) < 2: return
        sw, my = w / max(1, len(self.d) - 1), h / 2.0
        pr.setPen(QPen(QColor(BORDER_DIM), 1)); pr.drawLine(0, int(my), w, int(my))
        pr.setPen(QPen(QColor(NEON_RED), 1, Qt.DashLine)); pr.drawLine(0, int(my - h * 0.38), w, int(my - h * 0.38))
        pr.setPen(QPen(QColor(NEON_GREEN), 1, Qt.DashLine)); pr.drawLine(0, int(my + h * 0.38), w, int(my + h * 0.38))
        pts = [QPointF(i * sw, my - (max(min(v, 4.0), -4.0) / 4.0) * (h * 0.44)) for i, v in enumerate(self.d)]
        path = create_smooth_path(pts)
        fill = QPainterPath(path); fill.lineTo(w, my); fill.lineTo(0, my); fill.closeSubpath()
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(255, 0, 85, 70)); grad.setColorAt(0.5, QColor(249, 168, 3, 0)); grad.setColorAt(1.0, QColor(0, 255, 136, 70))
        pr.setBrush(grad); pr.setPen(Qt.NoPen); pr.drawPath(fill)
        pr.setPen(QPen(QColor(NEON_AMBER), 2)); pr.drawPath(path)

class MACDCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.macd, self.sig = deque(maxlen=40), deque(maxlen=40)
    def push(self, m, s):
        self.macd.append(m); self.sig.append(s)
        self.update()
    def paintEvent(self, e):
        pr = QPainter(self); pr.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pr.fillRect(0, 0, w, h, QColor(BG_BLACK))
        if len(self.macd) < 2: return
        my, sw = h / 2.0, w / max(1, len(self.macd) - 1)
        pr.setPen(QPen(QColor(BORDER_DIM), 1)); pr.drawLine(0, int(my), w, int(my))
        bw = max(sw * 0.6, 2)
        for i in range(len(self.macd)):
            hist = (self.macd[i] - self.sig[i]) * 15.0
            col = QColor(NEON_GREEN) if hist >= 0 else QColor(NEON_RED)
            pr.fillRect(int(i * sw - bw / 2), int(my if hist < 0 else my - hist), int(bw), max(int(abs(hist)), 1), col)
        pts_m = [QPointF(i * sw, my - self.macd[i] * 12.0) for i in range(len(self.macd))]
        pts_s = [QPointF(i * sw, my - self.sig[i] * 12.0) for i in range(len(self.sig))]
        pr.setPen(QPen(QColor(NEON_CYAN), 2)); pr.drawPath(create_smooth_path(pts_m))
        pr.setPen(QPen(QColor(NEON_AMBER), 1.5, Qt.DashLine)); pr.drawPath(create_smooth_path(pts_s))

class RSICanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.d = deque(maxlen=50)
    def push(self, v):
        self.d.append(v)
        self.update()
    def paintEvent(self, e):
        pr = QPainter(self); pr.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pr.fillRect(0, 0, w, h, QColor(BG_BLACK))
        if len(self.d) < 2: return
        sw, y70, y30 = w / max(1, len(self.d) - 1), h * 0.30, h * 0.70
        pr.setPen(QPen(QColor(BORDER_DIM), 1)); pr.drawLine(0, int(h * 0.5), w, int(h * 0.5))
        pr.setPen(QPen(QColor(NEON_RED), 1, Qt.DashLine)); pr.drawLine(0, int(y70), w, int(y70))
        pr.setPen(QPen(QColor(NEON_GREEN), 1, Qt.DashLine)); pr.drawLine(0, int(y30), w, int(y30))
        pts = [QPointF(i * sw, h * (1.0 - (v / 100.0))) for i, v in enumerate(self.d)]
        path = create_smooth_path(pts)
        fill = QPainterPath(path); fill.lineTo(w, h); fill.lineTo(0, h); fill.closeSubpath()
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(139, 92, 246, 80)); grad.setColorAt(1.0, QColor(139, 92, 246, 0))
        pr.setBrush(grad); pr.setPen(Qt.NoPen); pr.drawPath(fill)
        pr.setPen(QPen(QColor(NEON_LILAC), 2)); pr.drawPath(path)

class BollingerCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.upper, self.mid, self.lower = deque(maxlen=40), deque(maxlen=40), deque(maxlen=40)
    def push(self, u, m, l):
        self.upper.append(u); self.mid.append(m); self.lower.append(l)
        self.update()
    def paintEvent(self, e):
        pr = QPainter(self); pr.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pr.fillRect(0, 0, w, h, QColor(BG_BLACK))
        if len(self.upper) < 2: return
        sw = w / max(1, len(self.upper) - 1)
        min_v, max_v = min(min(self.lower), min(self.mid)), max(max(self.upper), max(self.mid))
        rng = max(max_v - min_v, 0.001)
        def ty(val): return h * (1.0 - (val - min_v) / rng)
        u_pts = [QPointF(i * sw, ty(self.upper[i])) for i in range(len(self.upper))]
        l_pts = [QPointF(i * sw, ty(self.lower[i])) for i in range(len(self.lower))]
        m_pts = [QPointF(i * sw, ty(self.mid[i])) for i in range(len(self.mid))]
        poly = QPainterPath(); poly.moveTo(u_pts[0])
        for p in u_pts[1:]: poly.lineTo(p)
        for p in reversed(l_pts): poly.lineTo(p)
        poly.closeSubpath()
        pr.setBrush(QColor(0, 240, 255, 30)); pr.setPen(Qt.NoPen); pr.drawPath(poly)
        pr.setPen(QPen(QColor(NEON_CYAN), 1.2)); pr.drawPath(create_smooth_path(u_pts))
        pr.setPen(QPen(QColor(NEON_AMBER), 1.5)); pr.drawPath(create_smooth_path(m_pts))
        pr.setPen(QPen(QColor(NEON_CYAN), 1.2)); pr.drawPath(create_smooth_path(l_pts))

class PressureCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.buy_p = 50.0
    def set_pressure(self, bp):
        self.buy_p = max(0.0, min(100.0, bp))
        self.update()
    def paintEvent(self, e):
        pr = QPainter(self); pr.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pr.fillRect(0, 0, w, h, QColor(BG_BLACK))
        bw = w * (self.buy_p / 100.0)
        pr.fillRect(0, int(h * 0.35), int(bw), int(h * 0.3), QColor(NEON_GREEN))
        pr.fillRect(int(bw), int(h * 0.35), int(w - bw), int(h * 0.3), QColor(NEON_RED))
        pr.setPen(QPen(QColor(TEXT_WHITE), 2))
        pr.drawLine(int(w * 0.5), int(h * 0.2), int(w * 0.5), int(h * 0.8))
        pr.setFont(QFont("Monospace", 8, QFont.Bold))
        pr.setPen(QColor(NEON_GREEN)); pr.drawText(6, int(h * 0.28), f"BUY {self.buy_p:.1f}%")
        pr.setPen(QColor(NEON_RED));   pr.drawText(int(w - 74), int(h * 0.28), f"SELL {100.0 - self.buy_p:.1f}%")

class WeightCanvas(QWidget):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn
    def paintEvent(self, e):
        pr = QPainter(self); pr.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pr.fillRect(0, 0, w, h, QColor(BG_BLACK))
        if not self.nn.wh[0]: return
        cols = [QColor(NEON_CYAN), QColor(NEON_AMBER), QColor(NEON_MAG), QColor(NEON_GREEN), QColor(NEON_RED), QColor(NEON_LILAC), QColor("#38BDF8"), QColor("#34D399")]
        nbls = ["Z", "RET", "VOL", "MACD", "RSI", "BB", "VWAP", "VOLAT"]
        sw = w / max(1, len(self.nn.wh[0]) - 1)
        for i in range(min(8, len(self.nn.wh))):
            d = list(self.nn.wh[i])
            mi, ma = min(d), max(d)
            rng = max(ma - mi, 0.0001)
            pts = [QPointF(j * sw, h - 10 - ((v - mi) / rng) * (h - 20)) for j, v in enumerate(d)]
            path = create_smooth_path(pts)
            pr.setPen(QPen(cols[i], 1.8)); pr.setBrush(Qt.NoBrush); pr.drawPath(path)
            if i % 2 == 0:
                pr.setFont(QFont("Monospace", 7, QFont.Bold))
                pr.drawText(w - 32, int(pts[-1].y() + 4), nbls[i])

class CompactAgentCard(QFrame):
    def __init__(self, name, c_hex):
        super().__init__()
        self.name, self.color, self.last_alpha = name, c_hex, 0.0
        self.setStyleSheet(f"background:{BG_CARD}; border:1px solid {BORDER_DIM}; border-radius:4px;")
        layout = QVBoxLayout(self); layout.setContentsMargins(8, 6, 8, 6); layout.setSpacing(2)
        top_row = QHBoxLayout()
        self.name_lbl = QLabel(name); self.name_lbl.setStyleSheet(f"color:{self.color}; font-size:11px; font-weight:bold; border:none;")
        self.health_lbl = QLabel("HLTH: 100%"); self.health_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:12px; font-weight:bold; border:none;")
        top_row.addWidget(self.name_lbl); top_row.addStretch(); top_row.addWidget(self.health_lbl)
        self.val_lbl = QLabel("+0.00%"); self.val_lbl.setStyleSheet(f"color:{TEXT_WHITE}; font-size:14px; font-weight:bold; border:none;")
        layout.addLayout(top_row); layout.addWidget(self.val_lbl)

    def push(self, alpha, health):
        self.health_lbl.setText(f"HLTH: {health:.0f}%")
        if abs(alpha - self.last_alpha) > 0.01:
            self.last_alpha = alpha
            self.val_lbl.setText(f"{alpha:+.2f}%")
            self.flash()

    def flash(self):
        self.setStyleSheet(f"background:{self.color}; border:1px solid {self.color}; border-radius:4px;")
        self.name_lbl.setStyleSheet("color:#000000; font-size:11px; font-weight:bold; border:none;")
        self.val_lbl.setStyleSheet("color:#000000; font-size:14px; font-weight:bold; border:none;")
        self.health_lbl.setStyleSheet("color:#000000; font-size:12px; font-weight:bold; border:none;")
        QTimer.singleShot(100, self.unflash)

    def unflash(self):
        self.setStyleSheet(f"background:{BG_CARD}; border:1px solid {BORDER_DIM}; border-radius:4px;")
        self.name_lbl.setStyleSheet(f"color:{self.color}; font-size:11px; font-weight:bold; border:none;")
        self.val_lbl.setStyleSheet(f"color:{TEXT_WHITE}; font-size:14px; font-weight:bold; border:none;")
        self.health_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:12px; font-weight:bold; border:none;")

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BONKAI // TICK-LEVEL NEURAL TRADING CORE")
        self.resize(1640, 960)
        self.setStyleSheet(f"background-color:{BG_BLACK}; color:{TEXT_WHITE}; font-family:Monospace;")
        self.bal, self.qty, self.ep, self.rpnl, self.upnl = 100000.0, 0.0, 0.0, 0.0, 0.0
        self.max_p, self.min_p = 0.0, float('inf')
        self.prices, self.px = deque(maxlen=500), None
        self.hist, self.act, self.cooldown, self.entry_t = [], False, 0, 0
        self.last_eval = time.time()
        self.ema12 = self.ema26 = None
        self.vwap_num = self.vwap_den = 0.0
        self.logger = HFTLogger()
        self.nn = FastMLP(input_size=8, hidden_size=16, weight_path="model_weights.npz")
        self.agents = []
        self.ui()
        self.w = MarketWorker()
        self.w.raw_tick.connect(self.tick)
        self.w.start()
        self.tmr = QTimer(self); self.tmr.timeout.connect(self.rndr); self.tmr.start(30)
        self.save_tmr = QTimer(self); self.save_tmr.timeout.connect(self.nn.save_weights); self.save_tmr.start(300000) 

    def closeEvent(self, e):
        self.nn.save_weights(); e.accept()

    def block(self, t, bc, tc):
        f = QFrame(); f.setStyleSheet(f"background-color:{BG_CARD}; border:1px solid {BORDER_DIM}; border-bottom: 2px solid {bc}; border-radius:4px;")
        l = QVBoxLayout(f); l.setContentsMargins(12, 8, 12, 8)
        t_l = QLabel(t); t_l.setStyleSheet(f"font-size:10px; color:{TEXT_MUTED}; font-weight:bold; border:none;")
        v_l = QLabel("--"); v_l.setStyleSheet(f"font-size:17px; font-weight:bold; color:{tc}; border:none;")
        l.addWidget(t_l); l.addWidget(v_l); f.v = v_l
        return f
        
    def panel(self, t, bc):
        f = QFrame(); f.setStyleSheet(f"background-color:{BG_CARD}; border:1px solid {BORDER_DIM}; border-top: 2px solid {bc}; border-radius:4px;")
        l = QVBoxLayout(f); l.setContentsMargins(8, 8, 8, 8)
        hl = QLabel(t); hl.setStyleSheet(f"font-size:11px; font-weight:bold; color:{bc}; border:none;")
        l.addWidget(hl)
        return f, l

    def ui(self):
        w = QWidget(); self.setCentralWidget(w)
        ml = QVBoxLayout(w); ml.setContentsMargins(12, 12, 12, 12); ml.setSpacing(10)
        h = QHBoxLayout()
        ttl = QLabel("BONKAI // MULTI-AGENT HFT TERMINAL"); ttl.setStyleSheet("font-size:20px; font-weight:bold; color:#FFFFFF; letter-spacing: 1px;")
        h.addWidget(ttl); h.addStretch()
        self.btn = QPushButton("ENGAGE SYSTEM"); self.btn.setStyleSheet(f"background:{NEON_GREEN}; color:#000000; font-size: 13px; font-weight:bold; padding:6px 22px; border-radius:4px;")
        self.btn.clicked.connect(self.tgl); h.addWidget(self.btn)
        ml.addLayout(h)
        tl = QHBoxLayout()
        self.c_eq, self.c_rp, self.c_up = self.block("WALLET EQUITY", NEON_CYAN, TEXT_WHITE), self.block("REALIZED P&L", NEON_GREEN, NEON_GREEN), self.block("UNREALIZED P&L", NEON_AMBER, NEON_AMBER)
        self.c_wr, self.c_ls, self.c_st = self.block("WIN RATE", NEON_MAG, NEON_MAG), self.block("NEURAL LOSS", NEON_RED, NEON_RED), self.block("STEPS", NEON_CYAN, NEON_CYAN)
        for x in [self.c_eq, self.c_rp, self.c_up, self.c_wr, self.c_ls, self.c_st]: tl.addWidget(x)
        ml.addLayout(tl)
        agent_box = QFrame(); agent_box.setStyleSheet(f"background-color:{BG_CARD}; border:1px solid {BORDER_DIM}; border-radius:4px;")
        abl = QHBoxLayout(agent_box); abl.setContentsMargins(6, 6, 6, 6); abl.setSpacing(6)
        names = [("CHINTU", NEON_CYAN), ("PINTU", NEON_AMBER), ("SHONA", NEON_MAG), ("MONA", NEON_GREEN), ("MACD", NEON_RED), ("VWAP", NEON_LILAC)]
        for n, c in names:
            ac = CompactAgentCard(n, c); self.agents.append(ac); abl.addWidget(ac)
        ml.addWidget(agent_box)
        split_h = QHBoxLayout()
        left_col = QVBoxLayout()
        self.mc = MarketCanvas(); f1, l1 = self.panel("ORDER DISPERSION & BROWNIAN CONE", NEON_CYAN); l1.addWidget(self.mc); left_col.addWidget(f1, 5)
        grid_plots = QGridLayout(); grid_plots.setSpacing(8)
        self.sc = SpectralCanvas(); f_sc, l_sc = self.panel("SPECTRAL IMBALANCE", NEON_MAG); l_sc.addWidget(self.sc); grid_plots.addWidget(f_sc, 0, 0)
        self.zc = ZScoreCanvas(); f_zc, l_zc = self.panel("MICROSTRUCTURE HESITATION", NEON_AMBER); l_zc.addWidget(self.zc); grid_plots.addWidget(f_zc, 0, 1)
        self.macd_plot = MACDCanvas(); f_mc, l_mc = self.panel("MACD DUAL MOMENTUM", NEON_RED); l_mc.addWidget(self.macd_plot); grid_plots.addWidget(f_mc, 1, 0)
        self.rsi_plot = RSICanvas(); f_rs, l_rs = self.panel("RSI OSCILLATION (30/70)", NEON_LILAC); l_rs.addWidget(self.rsi_plot); grid_plots.addWidget(f_rs, 1, 1)
        left_col.addLayout(grid_plots, 4); split_h.addLayout(left_col, 5)
        right_col = QVBoxLayout()
        bb_row = QHBoxLayout()
        self.bb_plot = BollingerCanvas(); f_bb, l_bb = self.panel("BOLLINGER VOLATILITY CORRIDOR", NEON_CYAN); l_bb.addWidget(self.bb_plot); bb_row.addWidget(f_bb, 1)
        self.press_plot = PressureCanvas(); f_pr, l_pr = self.panel("ORDER PRESSURE GAUGE", NEON_GREEN); l_pr.addWidget(self.press_plot); bb_row.addWidget(f_pr, 1)
        right_col.addLayout(bb_row, 2)
        self.wc = WeightCanvas(self.nn); f_wc, l_wc = self.panel("8D NEURAL WEIGHT EVOLUTION", NEON_AMBER); l_wc.addWidget(self.wc); right_col.addWidget(f_wc, 2)
        f6, l6 = self.panel("DIAGNOSTIC TRADE INSPECTOR", NEON_RED)
        self.tb = QTableWidget(0, 6)
        self.tb.setHorizontalHeaderLabels(["TIME", "TYPE", "PRICE", "REASON", "HOLD", "P&L"])
        self.tb.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.tb.verticalHeader().setVisible(False)
        self.tb.setStyleSheet(f"QTableWidget{{background:{BG_BLACK}; border:none; font-size:10px;}} QHeaderView::section{{background:{BG_CARD}; color:{TEXT_MUTED}; border:none; padding:4px; font-weight:bold;}}")
        l6.addWidget(self.tb); right_col.addWidget(f6, 3); split_h.addLayout(right_col, 4)
        ml.addLayout(split_h)

    def tgl(self):
        self.act = not self.act; self.cooldown = time.time() + 2.0
        self.btn.setText("HALT SYSTEM" if self.act else "ENGAGE SYSTEM")
        self.btn.setStyleSheet(f"background:{NEON_RED}; color:#FFFFFF; font-size:13px; font-weight:bold; padding:6px 22px; border-radius:4px;" if self.act else f"background:{NEON_GREEN}; color:#000000; font-size:13px; font-weight:bold; padding:6px 22px; border-radius:4px;")

    def tick(self, p, q, t, m):
        self.mc.push(p, q, m); self.prices.append(p); now = time.time()
        if self.ema12 is None:
            self.ema12 = self.ema26 = p; self.vwap_num = self.vwap_den = 0.0
        else:
            self.ema12, self.ema26 = self.ema12 * (1 - 2/13) + p * (2/13), self.ema26 * (1 - 2/27) + p * (2/27)
        self.vwap_num += p * q; self.vwap_den += q
        
        if len(self.prices) < 25 or now - self.last_eval < 1.0: return
        self.last_eval = now
        
        arr = np.array(self.prices)
        m_arr, s_arr = np.mean(arr[-50:]), np.std(arr[-50:])
        z, ret, vi = (p - m_arr) / (s_arr + 1e-6), (arr[-1] - arr[-10]) / (arr[-10] + 1e-6), (self.mc.vb - self.mc.vs) / ((self.mc.vb + self.mc.vs) + 1.0)
        macd = (self.ema12 - self.ema26) / (self.ema26 + 1e-6) * 100
        macd_sig = macd * 0.8
        diffs = np.diff(arr[-15:])
        rs = np.mean(np.maximum(diffs, 0)) / (np.mean(np.abs(np.minimum(diffs, 0))) + 1e-6)
        rsi_val, rsi_norm = 100 - (100 / (1 + rs)), (100 - (100 / (1 + rs)) - 50) / 50.0 
        upper, lower = m_arr + 2 * s_arr, m_arr - 2 * s_arr
        bb_norm = ((p - lower) / (upper - lower + 1e-6) - 0.5) * 2.0
        vwap = self.vwap_num / (self.vwap_den + 1e-6)
        vwap_dist, volat = (p - vwap) / (vwap + 1e-6) * 100, s_arr / (m_arr + 1e-6) * 100
        nx = np.array([z, ret * 100, vi, macd, rsi_norm, bb_norm, vwap_dist, volat])
        smooth_ret = (self.ema12 - self.ema26)
        if self.px is not None: self.nn.update(self.px, 1.0 if smooth_ret > 0 else -1.0)
        self.px = nx; sig = self.nn.predict(nx)
        
        self.sc.push(vi); self.zc.push(z); self.macd_plot.push(macd, macd_sig); self.rsi_plot.push(rsi_val); self.bb_plot.push(upper, m_arr, lower)
        tot_v = self.mc.vb + self.mc.vs
        self.press_plot.set_pressure((self.mc.vb / tot_v * 100) if tot_v > 0 else 50.0)
        
        mona_h = min(100.0, max(0.0, 100.0 - self.nn.loss * 100.0))
        self.agents[0].push(ret * 60, 100.0 - min(abs(ret) * 100, 50)); self.agents[1].push(-z * 10, 100.0 - min(abs(z) * 15, 60))
        self.agents[2].push(vi * 30, 100.0 - min(abs(vi) * 40, 50)); self.agents[3].push((1.0 - self.nn.loss) * 10, mona_h)
        self.agents[4].push(macd * 80, 85.0); self.agents[5].push(vwap_dist * 150, 90.0)
        
        if self.qty > 0: self.upnl = self.qty * (p - self.ep) * USD_TO_INR
        elif self.qty < 0: self.upnl = abs(self.qty) * (self.ep - p) * USD_TO_INR
        else: self.upnl = 0.0
        
        ca = "HOLD"
        if self.act:
            if self.qty == 0 and now > self.cooldown:
                if sig > 0.45:
                    a = self.bal * 0.25; self.qty, self.ep, self.bal = a / (p * USD_TO_INR), p, self.bal - a
                    self.max_p, self.entry_t, ca = p, now, "LONG"
                    self.log(t, "LONG", p, "MLP_BUY", "0s", 0)
                elif sig < -0.45:
                    a = self.bal * 0.25; self.qty, self.ep, self.bal = -(a / (p * USD_TO_INR)), p, self.bal - a
                    self.min_p, self.entry_t, ca = p, now, "SHORT"
                    self.log(t, "SHORT", p, "MLP_SELL", "0s", 0)
            elif self.qty > 0:
                self.max_p = max(self.max_p, p)
                pr, dd, rsn = (p - self.ep) / self.ep, (p - self.max_p) / self.max_p, ""
                if pr >= 0.003: rsn = "TAKE PROFIT"
                elif dd <= -0.0015: rsn = "TRAIL STOP"
                elif pr <= -0.0030: rsn = "HARD STOP"
                elif z < -3.5: rsn = "VOL SPIKE"
                elif sig < -0.4 and now - self.entry_t > 4.0: rsn = "REVERSAL"
                if rsn:
                    cst, val = self.qty * self.ep * USD_TO_INR, self.qty * p * USD_TO_INR
                    pnl = val - cst; self.bal += val; self.rpnl += pnl; self.hist.append(pnl)
                    self.log(t, "CLOSE", p, rsn, f"{int(now - self.entry_t)}s", pnl)
                    self.qty = self.ep = 0.0; self.cooldown = now + 4.0; ca = f"CLOSE_{rsn[:3]}"
            elif self.qty < 0:
                self.min_p = min(self.min_p, p)
                pr, dd, rsn = (self.ep - p) / self.ep, (self.min_p - p) / self.min_p, ""
                if pr >= 0.003: rsn = "TAKE PROFIT"
                elif dd <= -0.0015: rsn = "TRAIL STOP"
                elif pr <= -0.0030: rsn = "HARD STOP"
                elif z > 3.5: rsn = "VOL SPIKE"
                elif sig > 0.4 and now - self.entry_t > 4.0: rsn = "REVERSAL"
                if rsn:
                    cst, val = abs(self.qty) * self.ep * USD_TO_INR, abs(self.qty) * p * USD_TO_INR
                    pnl = cst - val; self.bal += (cst + pnl); self.rpnl += pnl; self.hist.append(pnl)
                    self.log(t, "COVER", p, rsn, f"{int(now - self.entry_t)}s", pnl)
                    self.qty = self.ep = 0.0; self.cooldown = now + 4.0; ca = f"COVER_{rsn[:3]}"
        self.logger.log_event(p, z, vi, macd, rsi_norm, vwap_dist, self.nn.loss, ca, self.rpnl)

    def log(self, t, s, p, rsn, h, pnl):
        self.tb.insertRow(0)
        for i, v in enumerate([t, s, f"${p:.1f}", rsn, h, f"₹{pnl:+.1f}" if s in ["CLOSE", "COVER"] else "OPEN"]):
            it = QTableWidgetItem(v); it.setTextAlignment(Qt.AlignCenter)
            if i == 1: it.setForeground(QColor(NEON_GREEN if s in ["LONG", "COVER"] else NEON_MAG))
            elif i == 3: it.setForeground(QColor(NEON_CYAN))
            elif i == 5 and s in ["CLOSE", "COVER"]: it.setForeground(QColor(NEON_GREEN if pnl >= 0 else NEON_RED))
            self.tb.setItem(0, i, it)
        if self.tb.rowCount() > 40: self.tb.removeRow(40)
        
    def rndr(self):
        self.mc.step(); self.wc.update()
        if self.qty > 0: eq = self.bal + (self.qty * self.mc.dp * USD_TO_INR)
        elif self.qty < 0: eq = self.bal + (abs(self.qty) * self.ep * USD_TO_INR) + (abs(self.qty) * (self.ep - self.mc.dp) * USD_TO_INR)
        else: eq = self.bal
        self.c_eq.v.setText(f"₹{eq:,.2f}")
        self.c_rp.v.setText(f"₹{self.rpnl:+.2f}"); self.c_rp.v.setStyleSheet(f"font-size:17px; font-weight:bold; border:none; color:{NEON_GREEN if self.rpnl >= 0 else NEON_RED};")
        self.c_up.v.setText(f"₹{self.upnl:+.2f}"); self.c_up.v.setStyleSheet(f"font-size:17px; font-weight:bold; border:none; color:{NEON_GREEN if self.upnl > 0 else NEON_RED if self.upnl < 0 else NEON_AMBER};")
        win_rate = (sum(1 for x in self.hist if x > 0) / len(self.hist) * 100) if self.hist else 0.0
        self.c_wr.v.setText(f"{win_rate:.1f}%"); self.c_ls.v.setText(f"{self.nn.loss:.4f}"); self.c_st.v.setText(f"{self.nn.steps:,}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = App(); w.show()
    sys.exit(app.exec_())
