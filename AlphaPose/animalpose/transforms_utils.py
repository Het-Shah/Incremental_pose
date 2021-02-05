import numpy as np

def get_max_pred(heatmaps):
  if not isinstance(heatmaps, np.ndarray):
    heatmaps = heatmaps.cpu().data.numpy()
  num_joints = heatmaps.shape[0]
  width = heatmaps.shape[1]
  heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
  idx = np.argmax(heatmaps_reshaped, 1)
  maxvals = np.max(heatmaps_reshaped, 1)

  maxvals = maxvals.reshape((num_joints, 1))
  idx = idx.reshape((num_joints, 1))

  preds = np.tile(idx, (1, 2)).astype(np.float32)

  preds[:, 0] = (preds[:, 0]) % width
  preds[:, 1] = np.floor((preds[:, 1]) / width)

  pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
  pred_mask = pred_mask.astype(np.float32)

  preds *= pred_mask
  return preds, maxvals
