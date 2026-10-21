set -u
cd /home/user/final/AVIRAT-V2
PY="/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/venv/bin/python -B"; E=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/eval; R=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/runs; V=S3c,Vta16,Vfa01
date +%T > $R/blockE_times.txt
$PY audit/motion/nav_eval.py . $E/nav_val_A0_default.csv $V > $R/nav_A0_default.log 2>&1; date +%T >> $R/blockE_times.txt
$PY audit/motion/nav_eval.py . $E/nav_val_B4_default.csv $V --set model.path=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/models/B4_official_s7.pt > $R/nav_B4_default.log 2>&1; date +%T >> $R/blockE_times.txt
$PY audit/motion/nav_eval.py . $E/nav_val_B4_decor.csv $V --set model.path=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/models/B4_official_s7.pt filter.motion_update_hz=1.0 filter.motion_err_tau_s=40 > $R/nav_B4_decor.log 2>&1; date +%T >> $R/blockE_times.txt
$PY audit/motion/nav_eval.py . $E/nav_val_B4_bv.csv $V --set model.path=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad/ml/models/B4_official_s7.pt filter.motion_bias_state=true filter.motion_bias_tau_s=40 > $R/nav_B4_bv.log 2>&1; date +%T >> $R/blockE_times.txt
echo done > $R/BLOCK_E_DONE
