P="/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/venv/bin/python -B -m src.train_motion_v2 --out /tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/models"
for e in B4 B5; do $P --exp $e > /tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/runs/${e}_official_s7.log 2>&1; done
for s in 11 13; do for e in A1 B1 B4 B5; do $P --exp $e --seed $s > /tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/runs/${e}_official_s$s.log 2>&1; done; done
for f in device_A route_Vw; do for e in A1 B1 B4 B5; do $P --exp $e --fold $f > /tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/runs/${e}_${f}.log 2>&1; done; done
echo done > /tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/runs/QUEUE_DONE
