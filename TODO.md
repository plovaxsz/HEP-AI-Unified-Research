# TODO - IPR Resume + evolution.log

- [x] Inspect `train.py` and identify missing `--resume`, per-epoch evolution logging, and fixed checkpoint logic.
- [ ] Implement `--resume` in `train.py`.
- [ ] Add `--run-id` + `evolution.log` append per epoch.
- [ ] Save checkpoint to `models/gatv2_final.pth` at end of every epoch (include optimizer/scheduler).
- [ ] Implement resume start_epoch and best_val_auc restoration.
- [ ] Run smoke test: 2 runs x 5 epochs with `--resume`; verify checkpoint + evolution.log.
- [ ] Provide/update a working `train_ipr.py` launcher script.


