#!/usr/bin/env bash
# Watches the Cholec80 focused-study + training, pushes milestones to Telegram.
cd /home/KHUser
N(){ ./telegram_notify.sh "$1" >/dev/null; }
FL=/tmp/tg_flags; mkdir -p "$FL"
once(){ [ -f "$FL/$1" ] && return 1; touch "$FL/$1"; return 0; }
FOCUS=/home/KHUser/sm_focused.log
TRAIN=/home/KHUser/sm_train.log

while true; do
  # SurgeNet training finished (pid 70735 gone)
  if ! kill -0 70735 2>/dev/null && once surgenet_done; then
    N "🏁 *Cholec80 SurgeNet 訓練完成*。協調器開始評估 + 訓練 control。"
  fi
  # crash / NaN in either log
  if grep -qE "Traceback|Grad NaN|out of memory" "$TRAIN" "$FOCUS" 2>/dev/null && once crash; then
    N "⚠️ *訓練偵測到錯誤* (Traceback/NaN/OOM)。我會去看 log 診斷。"
  fi
  # eval result blocks (SurgeNet then Control) — push each Accuracy line with its tag
  if grep -q "SurgeNet-surgical" "$FOCUS" 2>/dev/null && grep -A4 "SurgeNet-surgical" "$FOCUS" | grep -q "Accuracy" && once eval_surgenet; then
    acc=$(grep -A4 "SurgeNet-surgical" "$FOCUS" | grep Accuracy | head -1)
    N "📊 *SurgeNet backbone 評估完成*
$acc
(對照: 官方 ImageNet 重現 = 94.49%)
接著訓練 control (~1 天)..."
  fi
  if grep -q "Control-ImageNet" "$FOCUS" 2>/dev/null && grep -A4 "Control-ImageNet" "$FOCUS" | grep -q "Accuracy" && once eval_control; then
    acc=$(grep -A4 "Control-ImageNet" "$FOCUS" | grep Accuracy | head -1)
    N "📊 *Control (ImageNet) 評估完成*
$acc"
  fi
  # study done
  if grep -q "FOCUSED STUDY DONE" "$FOCUS" 2>/dev/null && once study_done; then
    s=$(grep -A2 "SurgeNet-surgical" "$FOCUS" | grep Accuracy | head -1)
    c=$(grep -A2 "Control-ImageNet" "$FOCUS" | grep Accuracy | head -1)
    N "✅ *聚焦研究全部完成*
SurgeNet: $s
Control : $c
→ 看 backbone 預訓練在 SOTA 時序頭下有沒有疊加。詳見 VM。"
    break
  fi
  sleep 90
done
