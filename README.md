# WR trainer

Open `train.ipynb` and run the setup and training cells. Training runs in a
managed Python worker, leaving the notebook available for other cells. Its
output appears below the training cell.

To stop all training and release the worker's GPU memory, run:

```python
trainer.stop_training()
```

This waits for training and its VPN/helper processes to exit. After 10 seconds
it forces termination if needed. To allow more time for graceful shutdown, use
`trainer.stop_training(timeout=30)`. It is safe to call the function repeatedly.
The worker owns its CUDA context, so exiting it releases its GPU allocations
and cached memory without restarting the notebook kernel. Other processes'
GPU allocations are unaffected.

`trainer.status()` returns `None` while the worker is running or before it has
started, and its exit code after it finishes. Zero means success. Rerun the
training cell to start again; it stops any previous worker first. In Python,
use `trainer.start(ROOT)` to start without blocking.

For a foreground terminal run, use `python train.py`; interrupt with Ctrl+C.
