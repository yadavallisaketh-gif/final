set -u
cd /home/user/final/AVIRAT-V2
PY="/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/venv/bin/python -B"
M=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/models; R=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/runs; E=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/eval
$PY -m src.train_motion_v2 --exp B5 --fold device_A --out $M > $R/B5_device_A.log 2>&1
for e in A1 B1 B4 B5; do $PY -m src.train_motion_v2 --exp $e --fold route_Vw --out $M > $R/${e}_route_Vw.log 2>&1; done
$PY -m src.train_motion_v2 --exp B1 --seed 11 --out $M > $R/B1_official_s11.log 2>&1
$PY audit/motion/run_eval.py . $E/official_val_v3.json official_val B1s11=$M/B1_official_s11.pt > $R/eval_official_s11.log 2>&1
$PY audit/motion/run_eval.py . $E/device_A.json device_A B5=$M/B5_device_A_s7.pt > $R/eval_device_A_B5.log 2>&1
$PY audit/motion/run_eval.py . $E/route_Vw.json route_Vw A0=A0 A1=$M/A1_route_Vw_s7.pt B1=$M/B1_route_Vw_s7.pt B4=$M/B4_route_Vw_s7.pt B5=$M/B5_route_Vw_s7.pt > $R/eval_route_Vw.log 2>&1
echo done > $R/BLOCK_B_DONE
